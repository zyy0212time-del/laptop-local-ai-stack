# The FreeToken Debugging / OSS Story

Two separate upstream interactions with **FreeToken** (FlashML-org, Apache-2.0).

## Bug 1 — decode stopped after two tokens (reported; fixed on the PR #131 branch)

Observed 2026-08-24 on Windows 11 + CUDA 13 + FreeToken 0.1.1(+PR #131):
loading a Qwen3.6-35B-A3B GGUF served fine (prefill healthy), but every
request returned exactly 2 completion tokens (`"user"`) and stopped. The same
GGUF decoded normally in llama.cpp.

Diagnosis (2026-08-24→26): the `untied lm_head` branch in
`qwen3_5_moe/gguf.py` returned logits for all prefill positions, while the
engine contract samples from the last position — so the first generated token
was sampled from position 0. A local fix (prefill slicing on `lm_head`) was
validated, then reported on FlashML-org/FreeToken PR #131; the author
reproduced it, confirmed the diagnosis ("Your diagnosis is right"), and fixed
it on the #131 PR branch in `b2f8475`. A second issue (MoE CUDA grid `z` 65535
cap, crashing long prefills) was fixed on the same PR branch in `9952a39`.

PR #131 remains **open and unmerged** at the last verified check (2026-09-07),
so neither commit has entered FlashML-org/FreeToken upstream `main`.

## Bug 2 — mixed-GGUF expert banks can't be sized (submitted patch, PR #1)

lucaspirola's Ornith branch (FlashML-org/FreeToken PR #196,
`ornith-sm120-gguf-mmq-phase2`) adds sm_120 dispatch + an int8-MMA MMQ port for
Ornith. Validating it on this exact RTX 5060 Laptop exposed a sizing gap:
`bank_bytes_estimate()` returned `None` for the mixed-GGUF route
(`expert_quant="gguf"`), so the automatic pageable-GPU residency planner
silently skipped split-residency planning. Mixed-GGUF checkpoints (llama.cpp
Q4_K_M, e.g. qwen35moe) use uniform flat `[num_experts, stride]` bank slots
whose strides are driven by the maximum participating quant geometry — raw
tensor payload bytes are not an equivalent substitute.

A fix was implemented and validated, then submitted as a pull request:

- **lucaspirola/FreeToken PR #1** — `fix(moe): size mixed-GGUF expert banks
  from GGUF metadata`
  - URL: https://github.com/lucaspirola/FreeToken/pull/1
  - state at last check (2026-09-01): **open**, not merged
  - one commit `c0673083`, **+163/−17**, 5 files, 8 CPU/synthetic regression
    tests
  - shared `expert_bank_geometry()` helper reused by the qwen35moe and Laguna
    GGUF loaders and by `bank_bytes_estimate()`
  - unknown/missing/malformed metadata returns `None` rather than a possibly
    unsafe estimate; non-GGUF estimation paths unchanged

Accurate role wording: I am the **external validator / debugger / patch author**
of that stacked PR on top of lucaspirola's branch. I am **not** the author of
FlashML-org/FreeToken PR #196. Neither PR was merged at the time of writing.

## Related

- FlashML-org/FreeToken PR #196 (author: lucaspirola): https://github.com/FlashML-org/FreeToken/pull/196
- FlashML-org/FreeToken PR #131 (author: vcruz305): https://github.com/FlashML-org/FreeToken/pull/131
