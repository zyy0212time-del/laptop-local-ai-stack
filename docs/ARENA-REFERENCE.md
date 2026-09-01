# Reasoning Budget Arena — Reference

The authoritative evidence repository is published at
<https://github.com/zyy0212time-del/reasoning-budget-arena>
(v1.0.0: https://github.com/zyy0212time-del/reasoning-budget-arena/releases/tag/v1.0.0).

## Protocol (both conditions)

- 6 models × 32 frozen questions (18 general + 14 cyber) = 192 requests per
  condition (384 model-question responses across Formal D + Formal C)
- llama.cpp b10375, one server at a time
- `ctx = 8192`, `max_tokens = 8192`, `temperature = 0.1`, `top_p = 0.9`
- no system prompt; vanilla chat templates
- Formal D: native/default thinking; Formal C: `--reasoning-budget 4096`
- blind final-only grading, scores locked before identity reveal

## Structural metrics (verified)

| | Formal D | Formal C |
|---|---|---|
| records | 192 | 192 |
| final | 119 | 192 |
| context-exhausted | 77 | 8 |
| confirmed loops | 0 | 6 |
| structurally clean | 115 | 184 |

## Overall blind scores /800

| model | D | C | Δ | rank D→C |
|---|---:|---:|---:|---|
| Ornith-1.5-35B-A3B | 569.5 | 725.5 | +156.0 | 1→2 |
| Nex-N2-mini | 353.0 | 746.5 | +393.5 | 4→1 |
| Gemma4-26B-A4B | 540.5 | 706.0 | +165.5 | 2→3 |
| Endy-Qwen3.6-CyberSec | 526.0 | 644.0 | +118.0 | 3→4 |
| Qwen3.8-9B-abliterated | 332.5 | 635.0 | +302.5 | 5→5 |
| RavenX-CyberAgent-35B | 301.5 | 576.0 | +274.5 | 6→6 |

## Why the ranking changed (documented, not claimed as causal)

- 0 empty finals in C vs 73 in D (failure mode shifted: → 6 loops + 2 truncated)
- every model scored higher in C; largest delta Nex +393.5
- two large deltas (Qwen 9B +302.5, RavenX +274.5) produced no rank change

See the arena repo's METHODOLOGY.md / RESULTS.md / LIMITATIONS.md for the
full context, including why this is an exploratory, closely-matched comparison
and not a strict controlled experiment.
