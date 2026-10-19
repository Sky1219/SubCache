**KV footprint** (K and V, d = 512, 8 layers)

| precision | per token per layer | per token, all layers | tokens of KV in 32 KB DTCM (all layers) | (one layer only) |
|---|---|---|---|---|
| int8 | 1024 B | 8.0 KiB | 4 | 32 |
| fp16 | 2048 B | 16.0 KiB | 2 | 16 |

**HyperRAM read traffic per decoded token** (int8 weights and KV, 90 MB/s)

| case | KV read | KV time | weights read | weight time | KV share | total ms/token |
|---|---|---|---|---|---|---|
| budget 16 | 136.0 KiB | 1.5 ms | 48.58 MiB | 566 ms | 0.3% | 568 |
| budget 32 | 264.0 KiB | 3.0 ms | 48.58 MiB | 566 ms | 0.5% | 569 |
| budget 64 | 520.0 KiB | 5.9 ms | 48.58 MiB | 566 ms | 1.0% | 572 |
| budget 128 | 1.01 MiB | 11.7 ms | 48.58 MiB | 566 ms | 2.0% | 578 |
| full cache, median story (201 tok) | 1.57 MiB | 18.3 ms | 48.58 MiB | 566 ms | 3.1% | 584 |
| full cache at ctx 2048 | 9.00 MiB | 104.9 ms | 48.58 MiB | 566 ms | 15.6% | 671 |

**Scorer cost per policy** at N = 128 slots per (layer, head), 8 layers x 16 heads

| policy | state / slot | extra buffering | ops per token per (layer, head) | scorer state, per-head keep sets | shared per layer | serial argmin, per-head |
|---|---|---|---|---|---|---|
| sliding window + sinks | 0 b | ring pointer per (layer, head) (11 bits) | - | 0 B | 0 B | - |
| H2O (float) / TIU (8-bit sat.) | 8 b | none | 129 add, 128 cmp | 16.1 KiB | 1.0 KiB | 16,512 cyc = 1.03 ms |
| EMA, alpha = 1 - 2^-k | 16 b | none (shift-subtract) | 258 add, 128 cmp | 32.2 KiB | 2.0 KiB | 16,512 cyc = 1.03 ms |
| EMA, general alpha (0.9, 0.99) | 16 b | none | 129 add, 129 mul, 128 cmp | 32.2 KiB | 2.0 KiB | 16,512 cyc = 1.03 ms |
| recent-window attention, W=16 | 8 b | last W attention rows (16 x N x 8 b) | 258 add, 128 cmp | 274.1 KiB | 17.1 KiB | 16,512 cyc = 1.03 ms |
