# Frontier Visual Reasoning Notes

This repo now has two visual semantic lanes:

1. `frontier VLM lane`
   - preferred when a local frontier checkpoint exists
   - intended for `Qwen2.5-VL`, `Florence-2`, or `Molmo2`
2. `semantic OpenCLIP lane`
   - local fallback when frontier checkpoints are absent

Current product doctrine:

- prefer the strongest local frontier open-weight VLM if available
- keep structural/operator grounding and verification alongside semantic captioning
- never pretend the fallback lane is frontier-level when it is not
- preserve RTX 4060 8GB viability with quantization and symbolic-first routing

Recommended local checkpoints for this repo:

- `Qwen2.5-VL-3B/7B` for general image and video reasoning
- `Florence-2-base-ft` for promptable captioning and grounding tasks
- `Molmo2-O-7B` for strong open multimodal scene understanding
