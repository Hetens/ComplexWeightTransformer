"""Environment sanity check: torch + CUDA reachability + complex autograd.

Run: uv run python scripts/check_env.py
Exits non-zero if torch or complex autograd is broken (CUDA absence is only a warning,
since CPU is a valid fallback for local smoke tests).
"""

from __future__ import annotations

import sys

import torch


def main() -> int:
    print(f"torch            : {torch.__version__}")
    print(f"cuda available   : {torch.cuda.is_available()}")
    print(f"compiled archs   : {torch.cuda.get_arch_list()}")

    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability(0)
        print(f"device           : {torch.cuda.get_device_name(0)}  sm_{cc[0]}{cc[1]}")
        x = torch.randn(512, 512, device="cuda")
        torch.cuda.synchronize()
        _ = (x @ x).sum().item()  # forces a real kernel launch (JIT warmup on Pascal)
        print("cuda matmul      : OK")
    else:
        print("device           : CPU (no CUDA) -- fine for smoke tests, use Unity for real runs")

    # Complex autograd: |Wz|^2 is a real scalar of complex params -> grads must be finite complex.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W = torch.randn(8, 6, dtype=torch.cfloat, device=dev, requires_grad=True)
    z = torch.randn(6, dtype=torch.cfloat, device=dev)
    out = W @ z
    loss = (out.conj() * out).real.sum()
    loss.backward()
    ok = W.grad is not None and W.grad.dtype == torch.complex64 and torch.isfinite(W.grad.abs()).all().item()
    print(f"complex autograd : {'OK' if ok else 'FAILED'}  (grad dtype={W.grad.dtype})")
    if not ok:
        return 1

    print("\nenvironment OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
