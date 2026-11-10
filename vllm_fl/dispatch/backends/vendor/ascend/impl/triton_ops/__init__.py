# Copyright (c) 2026 BAAI. All rights reserved.

"""
Triton kernels that replace a subset of the CANN fused operators.

Each module keeps a `triton_*` entry point that returns `None` when the input
layout is outside the supported set; the caller in `impl/` then keeps using the
original `torch_npu` operator, so the public function signature and the
numerical contract of the vendor path stay unchanged.
"""
