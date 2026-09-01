# Deployment Stack

The final deployment is a multi-model stack on llama.cpp (`llama-server`)
behind one routing layer, on a single consumer laptop. Each model runs in its
own server slot; only the active slot is resident at a time (32 GB RAM is the
hard constraint).

## Roles (maintainer decision, grounded in arena results)

| role | model | basis |
|---|---|---|
| Default / resident | Ornith-1.5-35B-A3B-Abliterated Q4_K_M | Formal D #1 (569.5/800); 43.5 t/s; ~3.5 GB VRAM |
| Hard tasks | Nex-N2-mini Q4_K_M | Formal C #1 (746.5/800); +393.5 under fixed 4096 budget |
| Secondary / different lineage | Gemma4-26B-A4B Q4_K_M | different base lineage + multimodal; previously the system's only resident |

## Runtime contract (from the arena records)

- llama.cpp b10375 (`ba360efe1`), `llama-server`
- `ctx 8192`, flash-attn on, `ngl 99`, `threads 24`
- MoE experts on CPU (`n_cpu_moe` = per-model expert count from GGUF metadata)
- `temperature 0.1`, `top_p 0.9` (capability runs)
- no system prompt (vanilla)
- reasoning policy: native by default; hard `--reasoning-budget 4096` available
  as a delivery lever (Formal C observed 0 empty finals in this run)

## Port/slot map (from the launcher, snapshot 2026-08-30)

| slot | model | notes |
|---|---|---|
| 30003 | Gemma4-26B | was the sole production resident |
| 30005 | Ornith-1.5-35B-A3B | arena Formal D winner |
| 30006 | Nex-N2-mini | arena Formal C winner |

## Memory behavior (measured)

- Ornith Q4_K_M: ~3564 MiB VRAM with experts on CPU
- Nex Q4_K_M: ~3537 MiB VRAM
- Gemma4-26B: ~5730 MiB VRAM (resident config)
- 8 GB VRAM cannot hold a 35B-A3B file (~20 GB); attention/dense on GPU,
  sparse experts on CPU is the usable daily configuration.

The reproducible experiment (questions, harness, objective/audit scripts,
score validation, figures) is published in the Reasoning Budget Arena
repository — this repo only documents the deployment decisions and their
evidence basis.
