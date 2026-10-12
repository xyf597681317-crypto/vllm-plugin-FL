# -*- coding: utf-8 -*-
"""最小设备健康检查：ones → ×2 → +1 校验和。用法：DSH_NPU=npu:12 python3 _probe_dev_health.py"""
import os

import torch
import torch_npu  # noqa: F401

dev = os.environ.get("DSH_NPU", "npu:0")
torch.npu.set_device(dev)
n = 1 << 20
x = torch.ones(n, dtype=torch.float32, device=dev)
y = (x * 2.0 + 1.0).sum().item()
exp = 3.0 * n
print(f"[health] {dev} sum={y} expect={exp} {'OK' if abs(y - exp) < 1 else 'BAD'}", flush=True)
