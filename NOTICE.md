# NOTICE

Attribution and provenance notice. This repository is an engineering case
study; it does not distribute model weights and claims no ownership over
third-party artifacts.

## Evaluated / deployed models (third-party GGUF fine-tunes)

| model | author/org | primary source |
|---|---|---|
| Ornith-1.5-35B-A3B-Abliterated | ornith-ai (base); mradermacher / codecraftersllc (abliterated GGUF suites) | huggingface.co/ornith-ai/Ornith-1.5-35B-A3B |
| Nex-N2-mini | nex-agi (original); bartowski (GGUF) | huggingface.co/nex-agi/Nex-N2-mini; huggingface.co/bartowski/nex-agi_Nex-N2-mini-GGUF |
| Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced | HauhauCS | huggingface.co/HauhauCS/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-MTP |
| RavenX-CyberAgent-35B-v5.1 | DeadByDawn101 | huggingface.co/deadbydawn101/RavenX-CyberAgent-Qwen3.6-35B-A3B-Opus-4.7-OpenMythos-Pentester-BugHunter-RATH-GGUF |
| Endy-Qwen3.6-CyberSec-35B-A3B | endystrike | huggingface.co/endystrike/Endy-Qwen3.6-CyberSec-35B-A3B |
| Qwen3.8-9B-abliterated-25 | base: QwenLM | github.com/QwenLM/Qwen3.8 (exact abliterated-25 GGUF source not recovered) |

Model names and metadata remain the responsibility of their upstream authors.
Exact local artifact fingerprints (SHA256) are recorded in the Reasoning
Budget Arena's `MODEL-ARTIFACT-MANIFEST.md` for provenance only.

## Runtime engines

- llama.cpp (`b10375`, build `ba360efe1`) — llama.cpp project, MIT, link-only.
- FreeToken (FlashML-org, Apache-2.0) — independent torch-based engine used
  as a separate runtime path; link-only.

## Base model families

- Qwen3.x (QwenLM, Alibaba) — Apache-2.0 (per Qwen3 series terms).
- Gemma 4 (Google) — Gemma license.
- Fine-tune/community card terms must be confirmed from each model's own card
  before any redistribution of its outputs.
