# -*- coding: utf-8 -*-
# BPE tokenizer + Two-Phase: Phase-1 base pretrain → save base.pt → Phase-2 LoRA finetune

import math, time
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
# === Matplotlib backend & compatibility ===
import os, sys
import matplotlib
# Prefer interactive backends; fallback to 'Agg' (save images) if none available
for _backend in ["TkAgg", "Qt5Agg", "QtAgg", "WXAgg"]:
    try:
        matplotlib.use(_backend, force=True)
        GUI_BACKEND_OK = True
        break
    except Exception:
        GUI_BACKEND_OK = False
if not GUI_BACKEND_OK:
    matplotlib.use("Agg", force=True)  # headless environment
import matplotlib.pyplot as plt

# =========================
#      Hyperparameters
# =========================
# Data & model
batch_size      = 32
block_size      = 128            # Recommend >=128 and <= max_ctx
n_embd          = 512
n_head          = 8
dropout_p       = 0.10
max_ctx         = 2048           # size of position embedding table
ENCODER_NAME    = "gpt2"  # you may change to "gpt2"

# Training steps & optimization
max_iters_phase1 = 1000          # steps for Phase-1 pretraining
max_iters_phase2 = 1000          # steps for Phase-2 LoRA finetuning
eval_interval    = 500
eval_iters       = 200
grad_accum       = 4             # gradient accumulation factor (effective batch ×4)

# Learning rate
lr_phase1_base = 1e-3            # base LR for Phase-1
lr_phase2_lora = 1e-3            # LR for Phase-2 (LoRA-only)
min_lr_ratio   = 0.1             # cosine anneal to base_lr*ratio

# ----- RevisitMemory (read-and-update) hyperparameters -----
mem_v_ref        = 1.0     # reference read voltage (no change to weights)
mem_eta          = 1e-4    # step size of weight drift per read (absolute scale)
mem_v_gain       = 0.25    # activity -> read-voltage offset gain
mem_target_act   = 1.0     # target activity (mean abs); >target → positive bias → weight increases
mem_wmin         = -1.0    # lower bound (device range)
mem_wmax         =  1.0    # upper bound

# LoRA hyperparameters
lora_r       = 8
lora_alpha   = 16
lora_dropout = 0.0               # recommend 0.0 for finetune

base_ckpt_path = "base.pt"
torch.manual_seed(1337)

# ----- Visualization for RevisitMemory -----
vis_enable   = True       # only show in Phase-2
vis_interval = 50         # refresh every N steps
vis_window   = "RevisitMemory (32x32)"


# =========================
#      Device (float32)
# =========================
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Using device:", device, "| AMP: False, GradClip: False")

# =========================
#        Data (BPE)
# =========================
enc = tiktoken.get_encoding(ENCODER_NAME)
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

data_ids = enc.encode(text)
vocab_size = enc.n_vocab
data = torch.tensor(data_ids, dtype=torch.long)

def decode(ids):
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return enc.decode(ids)

# train/val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

def get_batch(split):
    src = train_data if split == 'train' else val_data
    assert len(src) > block_size, "Text too short to sample; increase input.txt or reduce block_size"
    ix = torch.randint(len(src) - block_size, (batch_size,))
    x = torch.stack([src[i:i+block_size] for i in ix]).to(device, non_blocking=True)
    y = torch.stack([src[i+1:i+block_size+1] for i in ix]).to(device, non_blocking=True)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    losses = {}
    for split in ['train', 'val']:
        buf = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            buf[k] = loss
        losses[split] = buf.mean().item()
    model.train()
    return losses

# =========================
#   LR Scheduler: Warmup+Cosine
# =========================
def build_cosine_with_warmup(optimizer, warmup_steps, total_steps, base_lr, min_lr):
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cos = 0.5 * (1 + math.cos(math.pi * progress))
        target = min_lr + (base_lr - min_lr) * cos
        return target / base_lr
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# =========================
#     Base (Phase-1)
# =========================
class HeadBase(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout_p)
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x); q = self.query(x)
        wei = (q @ k.transpose(-2, -1)) * (q.size(-1) ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttentionBase(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([HeadBase(head_size) for _ in range(num_heads)])
        self.proj  = nn.Linear(num_heads * head_size, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout_p)
    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.proj(x)
        return self.dropout(x)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout_p),
        )
    def forward(self, x): return self.net(x)

class TransformerBlockBase(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        assert n_embd % n_head == 0
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa  = MultiHeadAttentionBase(n_head, head_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff  = FeedForward(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class BigramLanguageModelBase(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(max_ctx, n_embd)
        self.block = TransformerBlockBase(n_embd, n_head)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=True)
        # Weight tying (keep bias)
        self.lm_head.weight = self.token_embedding_table.weight
    def forward(self, idx, targets=None):
        idx = idx[:, -block_size:]
        B, T = idx.shape
        tok = self.token_embedding_table(idx)
        pos = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok + pos
        x = self.block(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            targets = targets[:, -block_size:]
            loss = F.cross_entropy(logits.reshape(B*T, vocab_size),
                                   targets.reshape(B*T))
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs  = F.softmax(logits, dim=-1)
            idx    = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx
class RevisitMemory(nn.Module):
    """
    Simulated "read-then-tune" memory array:
    - Hold a reference to an external weight tensor (nn.Parameter)
    - On each read(): use (v_read - v_ref) to apply a small in-place drift to weights
      w <- clamp(w + eta * (v_read - v_ref) * activity_scale, [wmin, wmax])
    """
    def __init__(self, weight_param: nn.Parameter,
                 v_ref=1.0, eta=1e-4, wmin=-float("inf"), wmax=float("inf")):
        super().__init__()
        self.weight = weight_param           # directly reference external Parameter (no copy)
        self.v_ref  = float(v_ref)
        self.eta    = float(eta)
        self.wmin   = float(wmin)
        self.wmax   = float(wmax)

    @torch.no_grad()
    def read_and_update(self, v_read: torch.Tensor,
                        activity_scale: torch.Tensor = None,
                        per_input_scale: torch.Tensor = None):
        """
        v_read            : scalar tensor
        activity_scale    : scalar tensor (global intensity such as mean |x|)
        per_input_scale   : [in_features] tensor; column-wise scaling for per-input weighting
        """
        if not torch.is_tensor(v_read):
            v_read = torch.tensor(float(v_read), dtype=self.weight.dtype, device=self.weight.device)
        delta = (v_read - self.v_ref)
        if activity_scale is not None:
            if not torch.is_tensor(activity_scale):
                activity_scale = torch.tensor(float(activity_scale), dtype=self.weight.dtype, device=self.weight.device)
            delta = delta * activity_scale

        if per_input_scale is not None:
            # Normalize to avoid explosion or all-zero
            s = per_input_scale
            if s.dim() != 1:  # [in_features]
                s = s.view(-1)
            s = s / (s.mean() + 1e-8)
            # Column-wise update: W += eta * delta * s[None, :]
            self.weight.data.add_(self.eta * delta * s.view(1, -1))
        else:
            # Legacy path: uniform offset for all elements
            self.weight.data.add_(self.eta * delta)

        self.weight.data.clamp_(self.wmin, self.wmax)


class MemLinear(nn.Module):
    """
    Linear layer with RevisitMemory (simulate "read-then-tune"):
      y = x @ W^T (+ b), and at each forward **read** the weight is slightly adjusted
      based on activity.
    - Parameter name 'mem_weight' to avoid interference with LoRA '.A.weight' / '.B.weight'
    """
    def __init__(self, in_features, out_features, bias=False,
                 v_ref=1.0, eta=1e-4, v_gain=0.25, target_activity=1.0,
                 wmin=-1.0, wmax=1.0):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))

        # Important: parameter name 'mem_weight' on purpose,
        # so it won't collide with LoRA '.A.weight' / '.B.weight' rules
        self.mem_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.mem_weight, a=math.sqrt(5))

        # Bind the "read-then-tune" memory
        self.memory = RevisitMemory(self.mem_weight, v_ref=v_ref, eta=eta, wmin=wmin, wmax=wmax)

        # Simple control to map "activity" to "read voltage"
        self.v_ref = float(v_ref)
        self.v_gain = float(v_gain)
        self.target_activity = float(target_activity)

    def _measure_activity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Use mean |x| over batch/time/feature dims as "activity" scalar.
        You can replace with more complex stats (e.g., L2 norm, quantiles).
        """
        # Support [B,T,C] or [B,C]
        dims = tuple(range(x.dim()))
        return x.abs().mean(dim=dims[1:]) if x.dim() >= 2 else x.abs().mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global scalar activity (legacy path)
        act = self._measure_activity(x)
        if act.dim() > 0:
            act = act.mean()

        # Read voltage
        v_read = self.v_ref + self.v_gain * (act - self.target_activity)

        # New: per-input activity (last dim is in_features)
        # Compatible with [B,T,C] or [B,C]: mean over all dims except last
        reduce_dims = tuple(range(x.dim() - 1))
        per_input = x.abs().mean(dim=reduce_dims)  # shape [in_features]

        # Read and update weights: with per_input_scale
        self.memory.read_and_update(
            v_read=v_read,
            activity_scale=act,
            per_input_scale=per_input
        )

        # Linear after read
        return F.linear(x, self.mem_weight, self.bias)
def _iter_mem_weights(module: nn.Module):
    """Iterate all MemLinear.mem_weight in the model (in order of appearance)."""
    for m in module.modules():
        if isinstance(m, MemLinear):
            yield m.mem_weight


def _get_mem_tile_32x32(model: nn.Module):
    """
    Collect all mem_weight -> flatten & concat -> take first 1024 -> 32x32 tile.
    If fewer than 1024, pad with zeros (rare).
    Return: np.ndarray [32,32]
    """
    with torch.no_grad():
        ws = []
        for w in _iter_mem_weights(model):
            ws.append(w.detach().reshape(-1))
        if not ws:
            # No MemLinear found (guard)
            return torch.zeros((32, 32), dtype=torch.float32).numpy()
        flat = torch.cat(ws)  # [N_total]
        if flat.numel() < 1024:
            pad = torch.zeros(1024 - flat.numel(), dtype=flat.dtype, device=flat.device)
            flat = torch.cat([flat, pad], dim=0)
        tile = flat[:1024].reshape(32, 32).to('cpu').numpy()
        return tile


def init_mem_visualizer(model_for_vis: nn.Module, vmin: float, vmax: float):
    """
    Initialize a figure, return (fig, ax, im, update_fn).
    update_fn(model) refreshes the image in-place and draws immediately.
    """
    plt.ion()
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    tile = _get_mem_tile_32x32(model_for_vis)
    im = ax.imshow(tile, vmin=vmin, vmax=vmax, interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("RevisitMemory Weights (32×32)")
    ax.set_xticks([]); ax.set_yticks([])

    def update_fn(model_current: nn.Module):
        new_tile = _get_mem_tile_32x32(model_current)
        im.set_data(new_tile)
        ax.set_title("RevisitMemory Weights (32×32)")  # you can add step text here
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    return fig, ax, im, update_fn


class MemVisualizer:
    """
    RevisitMemory 32×32 heatmap visualizer:
    - If GUI available: show and refresh in real time
    - Otherwise: save per-step PNG snapshots into ./mem_vis/
    """

    def __init__(self, model_for_vis: nn.Module, vmin: float, vmax: float, title="RevisitMemory (32x32)"):
        self.model = model_for_vis
        self.vmin = vmin
        self.vmax = vmax
        self.title = title
        self.use_gui = "Agg" not in matplotlib.get_backend()
        self.step = 0

        os.makedirs("mem_vis", exist_ok=True)

        if self.use_gui:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(4.6, 4.6))
            tile = self._get_tile_32x32()
            self.im = self.ax.imshow(tile, vmin=self.vmin, vmax=self.vmax, interpolation='nearest')
            self.cbar = self.fig.colorbar(self.im, ax=self.ax, fraction=0.046, pad=0.04)
            self.ax.set_title(self.title)
            self.ax.set_xticks([]);
            self.ax.set_yticks([])
            # Show without blocking and bring window to front if possible
            self.fig.canvas.draw_idle()
            try:
                self.fig.canvas.manager.show()
            except Exception:
                pass
            plt.show(block=False)
            plt.pause(0.001)  # let GUI event loop run

    @torch.no_grad()
    def _get_tile_32x32(self):
        ws = []
        for m in self.model.modules():
            if isinstance(m, MemLinear):
                w = m.mem_weight.detach().reshape(-1)
                ws.append(w)
        if not ws:
            return torch.zeros((32, 32), dtype=torch.float32).numpy()
        flat = torch.cat(ws)
        if flat.numel() < 1024:
            pad = torch.zeros(1024 - flat.numel(), dtype=flat.dtype, device=flat.device)
            flat = torch.cat([flat, pad], dim=0)
        return flat[:1024].reshape(32, 32).to('cpu').numpy()

    def update(self, step: int = None):
        """Call periodically within training loop."""
        if step is not None:
            self.step = step
        tile = self._get_tile_32x32()
        if self.use_gui:
            self.im.set_data(tile)
            self.ax.set_title(f"{self.title}  step={self.step}")
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.001)  # many envs need this to avoid blank window
        else:
            # Headless: save snapshot
            out = os.path.join("mem_vis", f"mem_tile_step_{self.step:06d}.png")
            plt.imsave(out, tile, vmin=self.vmin, vmax=self.vmax, format='png')

    def close(self):
        if self.use_gui:
            try:
                plt.ioff()
                plt.close(self.fig)
            except Exception:
                pass

from typing import Callable, Optional

class MemViewerCVPro:
    """
    Grayscale GUI (white=min, black=max) + Δ view:
    - Support "pin the same layer" 32x32 observation
    - transform_mode: "none" | "exp" (y=exp(exp_gain*x))
    - delta_mode: "vs_first" | "vs_prev"
    """

    def __init__(self, model_for_vis: nn.Module,
                 vmin: float = -1.0, vmax: float = 1.0,
                 window_name: str = "RevisitMemory (32x32)",
                 scale: int = 32,
                 phase_name: str = "P2",
                 total_steps: int = 2000,
                 delta_mode: str = "vs_prev",
                 vis_delta_eps: float = 1e-6,
                 layer_selector: Optional[Callable[[nn.Module], "MemLinear"]] = None,
                 transform_mode: str = "none",
                 exp_gain: float = 1.0,
                 # ↓↓↓ New: Δ display mode ("mag"=magnitude grayscale, "bw"=binary legacy)
                 delta_map: str = "mag",
                 **kwargs):
        self.model = model_for_vis
        self.vmin = float(vmin); self.vmax = float(vmax)
        self.win  = window_name
        self.scale = int(scale)
        self.phase = phase_name
        self.total_steps = int(total_steps)
        self.delta_mode = delta_mode
        self.eps = float(vis_delta_eps)

        # transform settings
        self.transform_mode = (transform_mode or "none").lower()
        self.exp_gain = float(exp_gain)
        self.delta_map = (delta_map or "mag").lower()

        # pin a specific layer (optional)
        self.target_layer = None
        if layer_selector is not None:
            try:
                self.target_layer = layer_selector(self.model)
            except Exception:
                self.target_layer = None

        # baseline frames (in transformed space, so Δ/visualization align)
        self.first_tile = None
        self.prev_tile  = None

        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        H = 32 * self.scale; W = 32 * self.scale
        spacer = int(W * 0.25); hud_h = 120
        self.canvas_w = W + spacer + W + 40
        self.canvas_h = H + hud_h + 40
        try:
            cv2.resizeWindow(self.win, self.canvas_w, self.canvas_h)
        except Exception:
            pass


    # ---------- core: weight read ----------
    @torch.no_grad()
    def _get_raw_tile_32x32(self):
        if self.target_layer is not None:
            w = self.target_layer.mem_weight.detach().reshape(-1)
            if w.numel() < 1024:
                pad = torch.zeros(1024 - w.numel(), dtype=w.dtype, device=w.device)
                flat = torch.cat([w, pad], dim=0)
            else:
                flat = w[:1024]
            return flat.reshape(32,32).to('cpu').numpy()

        ws = []
        for m in self.model.modules():
            if isinstance(m, MemLinear):
                ws.append(m.mem_weight.detach().reshape(-1))
        if not ws:
            tile = torch.zeros((32,32), dtype=torch.float32)
        else:
            flat = torch.cat(ws)
            if flat.numel() < 1024:
                pad = torch.zeros(1024 - flat.numel(), dtype=flat.dtype, device=flat.device)
                flat = torch.cat([flat, pad], dim=0)
            tile = flat[:1024].reshape(32,32)
        return tile.to('cpu').numpy()

    # ---------- transform ----------
    def _xfm(self, arr: np.ndarray) -> np.ndarray:
        if self.transform_mode == "exp":
            return np.exp(self.exp_gain * arr)
        return arr

    def _xfm_bounds(self, vmin: float, vmax: float) -> tuple:
        vv = np.array([vmin, vmax], dtype=np.float64)
        vv = self._xfm(vv)
        lo, hi = float(vv.min()), float(vv.max())
        if abs(hi - lo) < 1e-12:
            hi = lo + 1e-12
        return lo, hi

    # ---------- drawing & Δ ----------
    def _to_gray_min_white_max_black(self, arr: np.ndarray, lo: float = None, hi: float = None) -> np.ndarray:
        """
        Map array to grayscale: min→white, max→black, linear scale in between.
        If lo/hi not provided, use current frame min/max.
        """
        if lo is None or hi is None:
            lo, hi = float(arr.min()), float(arr.max())
        if abs(hi - lo) < 1e-12:
            hi = lo + 1e-12
        norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        gray = (255 * (1 - norm)).astype(np.uint8)  # black=large, white=small
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _delta_binary_bw(self, cur: np.ndarray, ref: np.ndarray):
        """
        Keep original function name for minimal changes; but support two modes:
        - self.delta_map == "mag": magnitude grayscale (per-frame minΔ→white, maxΔ→black)
        - else ("bw"): legacy binary threshold
        Return: (bgr_img, dmax, changed_count)
        """
        diff = np.abs(cur - ref)
        dmax = float(diff.max()) if diff.size else 0.0

        if self.delta_map == "mag":
            # Magnitude grayscale: min→white, max→black. Add epsilon to avoid divide-by-zero.
            dmin = float(diff.min()) if diff.size else 0.0
            rng = max(1e-12, dmax - dmin)
            norm = (diff - dmin) / rng  # [0,1]
            gray = (255 * (1 - norm)).astype(np.uint8)  # 0=black(large), 255=white(small)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            # "changed count" estimated with a tiny threshold (for stats only)
            changed = (diff > max(self.eps, dmin + 1e-12)).astype(np.uint8)
            changed_count = int(changed.sum())
            return img, dmax, changed_count

    # ===== legacy binary mode (backward compatible)=====
        changed = (diff > self.eps).astype(np.uint8)  # 1=changed
        changed_count = int(changed.sum())
        bw = (255 * (1 - changed)).astype(np.uint8)  # changed→0 black, unchanged→255 white
        return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR), dmax, changed_count

    def update(self, step:int, cur_lr:float=None, loss:float=None, avg_loss:float=None,
               eval_train:float=None, eval_val:float=None, **kwargs):
        # 1) get raw weights → 2) apply transform (if any) → 3) do grayscale/Δ in transformed space
        raw = self._get_raw_tile_32x32()
        tile = self._xfm(raw)

        # init baselines (in transformed space)
        if self.first_tile is None: self.first_tile = tile.copy()
        if self.prev_tile  is None: self.prev_tile  = tile.copy()

        # current grayscale (auto min/max on transformed tile)
        # lo, hi = self._xfm_bounds(self.vmin, self.vmax)
        left = self._to_gray_min_white_max_black(tile)


        # Δ: compare with previous or first frame (both in transformed space)
        ref = self.first_tile if (self.delta_mode == "vs_first") else self.prev_tile
        right, dmax, nchg = self._delta_binary_bw(tile, ref)

        # upscaling and layout
        H, W = left.shape[:2]
        left_big  = cv2.resize(left,  (W*self.scale, H*self.scale), interpolation=cv2.INTER_NEAREST)
        right_big = cv2.resize(right, (W*self.scale, H*self.scale), interpolation=cv2.INTER_NEAREST)
        spacer = np.ones((left_big.shape[0], int(left_big.shape[1]*0.25), 3), dtype=np.uint8) * 255
        panel = np.concatenate([left_big, spacer, right_big], axis=1)

        hud_h = 120; top_margin = 20; side_margin = 20
        h, w = panel.shape[:2]
        canvas_h = h + hud_h + 2*top_margin
        canvas_w = w + 2*side_margin
        canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

        y0 = top_margin + hud_h; x0 = side_margin
        canvas[y0:y0+h, x0:x0+w] = panel

        # HUD
        pct = 0.0 if self.total_steps <= 0 else min(1.0, step/self.total_steps)
        bar_w = int(w * 0.6); bar_h = 14; bx = x0; by = top_margin + 22
        cv2.rectangle(canvas, (bx, by), (bx + bar_w, by + bar_h), (40,40,40), 1)
        cv2.rectangle(canvas, (bx, by), (bx + int(bar_w*pct), by + bar_h), (0,0,0), -1)
        txt = (0,0,0)
        cv2.putText(canvas, f"{self.phase}   step {step}/{self.total_steps}  ({pct*100:.1f}%)",
                    (bx, by-6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, txt, 1, cv2.LINE_AA)
        ytxt = by + bar_h + 22
        if loss is not None and avg_loss is not None:
            cv2.putText(canvas, f"loss={loss:.5f}   avg={avg_loss:.5f}", (bx, ytxt),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, txt, 1, cv2.LINE_AA); ytxt += 22
        if cur_lr is not None:
            cv2.putText(canvas, f"lr={cur_lr:.2e}", (bx, ytxt),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, txt, 1, cv2.LINE_AA); ytxt += 22
        if eval_train is not None and eval_val is not None:
            cv2.putText(canvas, f"eval train={eval_train:.4f}   val={eval_val:.4f}", (bx, ytxt),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, txt, 1, cv2.LINE_AA); ytxt += 22

        # Show current transform & Δ stats
        cv2.putText(canvas, f"transform={self.transform_mode}(gain={self.exp_gain:.2f})   delta_mode={self.delta_mode}   eps={self.eps:.1e}   Δmax={dmax:.2e}   changed={nchg}",
                    (bx, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40,40,40), 1, cv2.LINE_AA)

        # Bottom labels
        cv2.putText(canvas, "CURRENT (min=white, max=black)", (x0, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (60,60,60), 1, cv2.LINE_AA)
        cv2.putText(canvas,
                    ("DELTA (magnitude: light=small, dark=large)"
                     if self.delta_map == "mag"
                     else "DELTA (black=changed, white=unchanged)"),
                    (x0 + w // 2 + spacer.shape[1] // 2 + 10, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (60, 60, 60), 1, cv2.LINE_AA)

        cv2.imshow(self.win, canvas)
        cv2.waitKey(1)

        # record previous frame (in transformed space)
        self.prev_tile = tile.copy()

    def close(self):
        try:
            cv2.destroyWindow(self.win)
        except Exception:
            pass

def pick_head0_query_A(model: nn.Module):
    """
    Prefer returning head0.query.A (MemLinear); if not present due to structure differences,
    fall back to the first MemLinear in the model.
    """
    # 1) try canonical path
    try:
        tl = model.block.sa.heads[0].query.A
        if isinstance(tl, MemLinear) and hasattr(tl, "mem_weight"):
            return tl
    except Exception:
        pass

    # 2) fallback by iteration
    for m in model.modules():
        if isinstance(m, MemLinear) and hasattr(m, "mem_weight"):
            return m
    raise RuntimeError("No observable MemLinear found (check LoRA injection and module path).")

# =========================
#     LoRA (Phase-2)
# =========================
class LoRALinear(nn.Module):
    # y = base(x) + scale * B(A(dropout(x)))
    def __init__(self, in_features, out_features, r=8, alpha=16, dropout=0.0, bias=False):
        super().__init__()
        self.r = r
        self.scale = (alpha / r) if r > 0 else 0.0

        # Base weight (loaded from base.pt), not part of "read-then-tune"
        self.base = nn.Linear(in_features, out_features, bias=bias)

        if r > 0:
            # Replace original nn.Linear (A/B) with MemLinear integrating RevisitMemory
            self.A = MemLinear(
                in_features, r, bias=False,
                v_ref=mem_v_ref, eta=mem_eta, v_gain=mem_v_gain,
                target_activity=mem_target_act, wmin=mem_wmin, wmax=mem_wmax
            )
            self.B = MemLinear(
                r, out_features, bias=False,
                v_ref=mem_v_ref, eta=mem_eta, v_gain=mem_v_gain,
                target_activity=mem_target_act, wmin=mem_wmin, wmax=mem_wmax
            )
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        else:
            self.A = None
            self.B = None
            self.dropout = nn.Identity()

    def forward(self, x):
        out = self.base(x)
        if self.r > 0:
            # A/B will lightly self-update their weights at forward time due to "read voltage"
            low_rank = self.B(self.A(self.dropout(x)))
            out = out + self.scale * low_rank
        return out


class HeadLoRA(nn.Module):
    def __init__(self, head_size, lora_r=8, lora_alpha=16, lora_dropout=0.0):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)  # loaded directly
        self.query = LoRALinear(n_embd, head_size, r=lora_r, alpha=lora_alpha, dropout=lora_dropout, bias=False)
        self.value = LoRALinear(n_embd, head_size, r=lora_r, alpha=lora_alpha, dropout=lora_dropout, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout_p)
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x); q = self.query(x)
        wei = (q @ k.transpose(-2, -1)) * (q.size(-1) ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttentionLoRA(nn.Module):
    def __init__(self, num_heads, head_size, lora_r=8, lora_alpha=16, lora_dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList([
            HeadLoRA(head_size, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            for _ in range(num_heads)
        ])
        self.proj  = nn.Linear(num_heads * head_size, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout_p)
    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.proj(x)
        return self.dropout(x)

class TransformerBlockLoRA(nn.Module):
    def __init__(self, n_embd, n_head, lora_r=8, lora_alpha=16, lora_dropout=0.0):
        super().__init__()
        head_size = n_embd // n_head
        assert n_embd % n_head == 0
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa  = MultiHeadAttentionLoRA(n_head, head_size, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff  = FeedForward(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class BigramLanguageModelLoRA(nn.Module):
    def __init__(self, vocab_size, lora_r=8, lora_alpha=16, lora_dropout=0.0):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(max_ctx, n_embd)
        self.block = TransformerBlockLoRA(n_embd, n_head, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=True)  # align with base (with bias)
        # Weight tying as in base for stability
        self.lm_head.weight = self.token_embedding_table.weight
    def forward(self, idx, targets=None):
        idx = idx[:, -block_size:]
        B, T = idx.shape
        tok = self.token_embedding_table(idx)
        pos = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok + pos
        x = self.block(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            targets = targets[:, -block_size:]
            loss = F.cross_entropy(logits.reshape(B*T, vocab_size),
                                   targets.reshape(B*T))
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs  = F.softmax(logits, dim=-1)
            idx    = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx

# =========================
#    Mapping & Freezing
# =========================
def load_base_into_lora(lora_model: nn.Module, base_sd: dict):
    """Map base weights into LoRA model:
       ...query.weight -> ...query.base.weight
       ...value.weight -> ...value.base.weight
       Others keep original names (embedding, key, proj, FFN, LN, lm_head, etc.)"""
    lora_sd = lora_model.state_dict()
    mapped = {}
    for k, v in base_sd.items():
        new_k = k
        if ".sa.heads." in k and k.endswith("query.weight"):
            new_k = k.replace("query.weight", "query.base.weight")
        elif ".sa.heads." in k and k.endswith("value.weight"):
            new_k = k.replace("value.weight", "value.base.weight")
        if new_k in lora_sd and lora_sd[new_k].shape == v.shape:
            mapped[new_k] = v
    lora_sd.update(mapped)
    lora_model.load_state_dict(lora_sd, strict=False)
    print(f"[LoRA Load] mapped {len(mapped)} tensors from base → LoRA.base")

def freeze_to_lora_only(model: nn.Module, train_ln=True, unfreeze_head_embed=False):
    for _, p in model.named_parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if ('.A.weight' in name) or ('.B.weight' in name):
            p.requires_grad = True
        elif train_ln and ('ln' in name) and (('weight' in name) or ('bias' in name)):
            p.requires_grad = True
    if unfreeze_head_embed:
        model.token_embedding_table.weight.requires_grad = True
        model.lm_head.weight.requires_grad = True
        if model.lm_head.bias is not None:
            model.lm_head.bias.requires_grad = True

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# =========================
#         Train Utils
# =========================
import sys

from functools import partial

# Unified: tqdm outputs to stdout; disable colors (older tqdm might ignore colour arg; stdout also avoids "red text")
TQDM = partial(tqdm, file=sys.stdout, colour=None)

def train_loop(model, optimizer, scheduler, max_steps, phase_name="P", get_batch_fn=get_batch):
    model.train()
    t0 = time.time()
    avg_loss = None
    # —— P2 visualization ——
    mem_viewer = None
    last_eval = {"train": None, "val": None}
    if vis_enable and phase_name.startswith("P2"):
        mem_viewer = MemViewerCVPro(
            model_for_vis=lora_model, vmin=mem_wmin, vmax=mem_wmax,
            window_name=vis_window, scale=32,
            phase_name=phase_name, total_steps=max_steps,
            delta_mode="vs_prev", vis_delta_eps=1e-6,
            layer_selector=pick_head0_query_A,  # keep "pinning the same tile"
            transform_mode="exp", exp_gain=3.0,
            delta_map="mag"  # magnitude grayscale for Δ
        )

    # Single progress bar; to stdout; no colors
    with TQDM(
        total=max_steps,
        desc=f"{phase_name} Training",
        ncols=100,
        leave=True,
        position=0,
        mininterval=0.2,  # refresh frequency control
        miniters=1
    ) as pbar:
        for step in range(1, max_steps + 1):
            optimizer.zero_grad(set_to_none=True)

            # gradient accumulation
            for _ in range(grad_accum):
                xb, yb = get_batch_fn('train')
                _, loss = model(xb, yb)
                (loss / grad_accum).backward()

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # update bar (won't create a new one)
            loss_val = float(loss.detach())
            avg_loss = loss_val if avg_loss is None else (0.9 * avg_loss + 0.1 * loss_val)
            cur_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(loss=f"{loss_val:.4f}", avg=f"{avg_loss:.4f}", lr=f"{cur_lr:.2e}")
            pbar.update(1)
            # —— refresh GUI ——
            if mem_viewer is not None and (step % vis_interval == 0 or step == 1):
                mem_viewer.update(
                    step=step, cur_lr=cur_lr, loss=loss_val, avg_loss=avg_loss,
                    eval_train=last_eval["train"], eval_val=last_eval["val"]
                )

            # evaluation logs: use pbar.write to stdout; avoid print
            if step % eval_interval == 0 or step == 1:
                stats = estimate_loss(model)
                dt = time.time() - t0;
                t0 = time.time()
                last_eval["train"] = stats["train"]
                last_eval["val"] = stats["val"]
                pbar.write(
                    f"[{phase_name}] step {step:5d} | lr {cur_lr:.2e} | "
                    f"train {stats['train']:.4f} | val {stats['val']:.4f} | {dt:.1f}s",
                    file=sys.stdout
                )
                pbar.refresh()

    if mem_viewer is not None:
        mem_viewer.close()

    # final message to stdout as well
    tqdm.write(f"{phase_name} finished. Final avg loss {avg_loss:.4f}", file=sys.stdout)

@torch.no_grad()
def sample_and_print(model, prefix=""):
    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    gen = model.generate(start, 200)[0].to('cpu')
    print(prefix + decode(gen))

# =========================
#        Phase-1
# =========================
print("\n=== Phase-1: Base pretrain (BPE, fp32) ===")
base = BigramLanguageModelBase(vocab_size).to(device)
opt1  = torch.optim.AdamW(base.parameters(), lr=lr_phase1_base, betas=(0.9, 0.95), weight_decay=0.01)
sch1  = build_cosine_with_warmup(opt1, warmup_steps=200, total_steps=max_iters_phase1,
                                 base_lr=lr_phase1_base, min_lr=lr_phase1_base*min_lr_ratio)

train_loop(base, opt1, sch1, max_steps=max_iters_phase1, phase_name="P1")
torch.save(base.state_dict(), base_ckpt_path)
print(f"Saved base weights → {base_ckpt_path}")
sample_and_print(base, prefix="\n[Base sample] ")

# =========================
#        Phase-2
# =========================
print("\n=== Phase-2: LoRA finetune (fp32) ===")
lora_model = BigramLanguageModelLoRA(vocab_size, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout).to(device)
base_sd = torch.load(base_ckpt_path, map_location="cpu")
load_base_into_lora(lora_model, base_sd)

# Freeze: train LoRA + LN only; for lower loss you may set unfreeze_head_embed=True (with smaller LR)
freeze_to_lora_only(lora_model, train_ln=True, unfreeze_head_embed=True)

# Optimizer parameter groups: LoRA / LN / head & embeddings
lora_params, ln_params, head_embed = [], [], []
for n,p in lora_model.named_parameters():
    if not p.requires_grad: continue
    if ('.A.weight' in n) or ('.B.weight' in n):
        lora_params.append(p)
    elif ('ln' in n):
        ln_params.append(p)
    else:
        head_embed.append(p)  # token_embedding, lm_head weights/bias

opt2 = torch.optim.AdamW(
    [
        {'params': lora_params,  'lr': lr_phase2_lora,         'weight_decay': 0.0},
        {'params': ln_params,    'lr': lr_phase2_lora*0.5,     'weight_decay': 0.0},
        {'params': head_embed,   'lr': lr_phase2_lora*0.25,    'weight_decay': 0.01},
    ],
    betas=(0.9, 0.999), eps=1e-8
)
sch2 = build_cosine_with_warmup(opt2, warmup_steps=100, total_steps=max_iters_phase2,
                                base_lr=lr_phase2_lora, min_lr=lr_phase2_lora*min_lr_ratio)

total_params, trainable_params_cnt = count_params(lora_model)
print(f"Total params: {total_params/1e6:.3f}M | Trainable: {trainable_params_cnt/1e6:.3f}M")

train_loop(lora_model, opt2, sch2, max_steps=max_iters_phase2, phase_name="P2")
sample_and_print(lora_model, prefix="\n[LoRA sample] ")
print("\nDone.")
