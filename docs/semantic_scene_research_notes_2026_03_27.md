# Semantic Scene Research Notes

This repo's new semantic scene-understanding lane is guided by recent multimodal and vision research, adapted for a local `RTX 4060 8GB` environment.

Primary reference families:

- `Qwen2.5-VL`
  - strong image/video reasoning
  - visual localization and structured outputs
  - reference: <https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct>
- `Florence-2`
  - promptable unified vision tasks such as captioning, dense region grounding, OCR, and phrase grounding
  - reference: <https://huggingface.co/microsoft/Florence-2-base-ft>
- `Grounding DINO`
  - open-vocabulary object localization
  - reference: <https://github.com/IDEA-Research/GroundingDINO>
- `SAM 2`
  - segmentation and streaming memory for images/videos
  - reference: <https://arxiv.org/abs/2408.00714>
- `DINOv2`
  - robust all-purpose visual features
  - reference: <https://arxiv.org/abs/2304.07193>
- `SigLIP 2`
  - improved semantic understanding, localization, and dense features
  - reference: <https://huggingface.co/google/siglip2-large-patch16-256>
- `Molmo`
  - strong open multimodal reasoning and screen/image understanding
  - reference: <https://huggingface.co/allenai/Molmo-7B-D-0924>

Implementation choice in this repo:

- keep the existing operator-algebra / structural parser
- add a semantic lane on top instead of replacing the structural lane
- use local OpenCLIP-style zero-shot semantic scoring for scene, object, overlay, and region labels
- keep answers honest when semantic grounding is weak
- surface `scene semantic level`, semantic caption, and backend in the GUI

This is not a full reproduction of large frontier VLMs. It is a local, product-oriented adaptation that upgrades general image scene understanding while preserving:

- verification
- grounding transparency
- RTX 4060 compatibility
- beginner-friendly UX
