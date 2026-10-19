# CPU vs GPU reproduction

Configs compared: 140 (only in one run: 0).

| metric | largest |CPU - GPU| over all configs |
|---|---|
| recall-token top-1 accuracy | 0.154 points |
| recall-token loss (nats) | 0.0043 nats |
| name-token top-1 accuracy | 0.323 points |
| perplexity | 0.038% (relative) |
| top-1 agreement with full cache | 0.026 points |
| KL(full || policy), nats/token | 0.0004 nats |
| next-token top-1 accuracy | 0.035 points |

Per-story count of correct top-1 predictions identical on both devices: 99.2% of (config, story) pairs.
