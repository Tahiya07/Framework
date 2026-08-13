# CPU deployment benchmark

CPU: `Intel64 Family 6 Model 154 Stepping 4, GenuineIntel`  
Torch threads: `4`

| Variant | Status | Size (MiB) | Load (s) | RSS Δ (MiB) | p50 (ms) | p95 (ms) | Req/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| merged (merged_fp32) | ok | 1899.78 | 7.718 | 57.94 | 1161.27 | 1245.50 | 0.849 |
| lightweight (fp16_merged) | ok | 957.48 | 3.920 | -193.23 | 290859.23 | 339460.97 | 0.004 |

## Deployment decision

Use the smallest **accuracy-validated** variant with acceptable p95 latency and RSS. For this repository, dynamic PyTorch INT8 is explicitly rejected by `quantize_bloom.py` because it caused major accuracy loss. ONNX Runtime INT8 is a promising offline-CPU candidate, but only deploy it after matching classifier predictions/metrics against the merged FP32 checkpoint.
