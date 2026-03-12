# QLoRA and Distillation Roadmap

## Conclusion

For this project, QLoRA is the enabler and distillation is the reasoning booster.

- QLoRA makes training feasible on an RTX 4060 8GB.
- Distillation improves reasoning only if the teacher traces contain the right intermediate structure.
- The best target is not raw final answers but structured traces:
  - hidden goals
  - required and missing premises
  - operator decomposition
  - goal-preservation judgments
  - grounded visual relations

## Why this fits SemOp

SemOp already stores intermediate reasoning artifacts in code:

- `StructuredMeaningPipeline` exposes hidden-premise and operator traces.
- `CompetitiveProgrammingReasoner` exposes DSL/frame/algorithm structure.
- `VLSOReasoner` exposes grounded entities, relations, operators, and answer evidence.

That means distillation can train the student on world-model construction, not only on final responses.

## Recommended training order

1. Hidden-premise distillation
   - Train a small text model to predict hidden goals, required premises, missing premises, and clarification score.
2. CP structuring distillation
   - Train a parser that maps problem statements to frames, DSL operators, and algorithm family.
3. VLSO grounded-QA distillation
   - Distill grounded entities, relations, operators, and grounded answer policy.
4. Joint multi-task distillation
   - Mix the three trace families into a single SFT dataset.

## Trace schema

Teacher traces are exported as JSONL records with this shape:

- `task`
- `input_text`
- `input_payload`
- `teacher_trace`
- `completion_payload`
- `metadata`

Supported tasks:

- `hidden_premise`
- `cp_structuring`
- `vlso_grounded_qa`

## Export command

```bash
.\.venv312\Scripts\python.exe tools\eval\export_teacher_traces.py ^
  --hidden-premises examples\hidden_premise_eval.jsonl ^
  --cp-input examples\cp_parser_eval.jsonl ^
  --vlso-input examples\vlso_eval.jsonl ^
  --examples-root examples ^
  --output data\teacher_traces.jsonl ^
  --sft-output data\teacher_traces_sft.jsonl
```

## RTX 4060 8GB recommendations

- Base model size: `0.5B` to `1.5B`
- Method: `QLoRA / LoRA`
- Sequence length: `512`
- Batch size: `1`
- Gradient accumulation: `8` to `16`
- Prefer `fp16` or `bf16` if available

## Why not full multimodal finetuning first

For this project, end-to-end multimodal finetuning is not the first bottleneck. The bottleneck is whether the model can reconstruct:

- hidden premises
- operator relations
- grounded constraints
- structured plans

That is why trace distillation is the correct first step.

## External references

- [QLoRA](https://huggingface.co/papers/2305.14314)
- [Distilling Step-by-Step](https://huggingface.co/papers/2305.02301)
- [PEFT quantization guide](https://huggingface.co/docs/peft/main/en/developer_guides/quantization)

For a harder distillation set, include `examples\cp_hidden_constraint_eval.jsonl` and `examples\vlso_real_image_eval_gold.jsonl` in the export command.
