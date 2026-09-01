# Runtime Evidence

Machine-specific measured numbers referenced by the README. Every value below
was recorded in the local project artifacts (audit + bench logs) and is
reproducible from the Reasoning Budget Arena repository where applicable.

## Hardware (machine audit, 2026-08-13)

| item | value |
|---|---|
| CPU | Intel Core Ultra 9 275HX, 24 cores, 2.7 GHz |
| GPU | NVIDIA GeForce RTX 5060 Laptop, 8 GB (8151 MiB) VRAM, sm_120 |
| RAM | 32 GB DDR5 (2×16 GB, 5600 MHz) |
| SSD | 1 TB NVMe, measured ~2158 MB/s sequential read |
| OS | Windows 11, Build 26200 |

## llama.cpp benchmark — experts on GPU vs CPU (same file)

From the archived `llama-bench` records on the same GPU/runtime:

| model | config | tg64 t/s |
|---|---|---|
| qwen35moe 35B.A3B Q4_K_M (18.23 GiB) | `ngl=99`, experts on GPU | **4.15** |
| qwen35moe 35B.A3B Q4_K_M | `ngl=99`, `n_cpu_moe=256` (experts on CPU) | **43.31** |

## Arena speed gate (llama.cpp b10375, ctx 8192, ngl=99, experts on CPU)

| model | gen t/s | TTFT |
|---|---:|---:|
| RavenX-CyberAgent-35B-v5.1 | 44.8 | 5.092 s |
| Endy-Qwen3.6-CyberSec-35B-A3B | 45.2 | 2.908 s |
| Gemma4-26B-A4B-Balanced | 37.7 | not measured |
| Ornith-1.5-35B-A3B | 43.5 | not measured |
| Nex-N2-mini | 47.0 | not measured |
| Qwen3.8-9B-abliterated-25 | 58.2 | not measured |

All six passed the ≥10 t/s gate. Speed was scored separately and never folded
into answer quality.

## FreeToken backend A/B (Qwen3.6-35B-A3B MTP unified Q4, same engine)

| backend | decode t/s | TTFT |
|---|---|---|
| `--moe-backend cpu` | 11.2 / 14.1 / 14.6 | 4.2–11.5 s |
| offload + `--moe-cache-auto` | ~51 / ~54 / ~51 | 1.6–3.7 s (hot cache 0.79–0.88 s) |
| hybrid | 15.7 / 23.0 / 18.1 | 2.7–5.0 s |

## Mixed-GGUF expert-bank sizing (patch validation, Ornith Q4_K_M)

- gate/up stride: `1,179,648` B/expert
- down stride: `860,160` B/expert
- logical bank size per layer: `522,190,848` B
- 40-layer total: `20,887,633,920` B
- estimator delta vs actual geometry: **0**
- E2E (RTX 5060 Laptop): 15 GiB pin budget → 10 pageable + 30 pinned MoE
  layers, 60/60 host registrations, `14.589844` GiB cumulatively registered,
  `/health = ready`, HTTP 200 `"OK"`

## Runtime relationship

Two independent runtime paths were used at different stages, both consuming
the same GGUF files:

- **Runtime path A — llama.cpp (`llama-server`)**: production resident model
  and the canonical arena runtime for all six models.
- **Runtime path B — FreeToken (`ft.exe`, FlashML-org)**: separate torch-based
  engine used for the earlier uniform-quant 35B serving; rejected mixed-GGUF
  expert banks, which is why it was never used for Ornith/Nex.

No claim is made that one runtime is a fork of, or delegates to, the other.
