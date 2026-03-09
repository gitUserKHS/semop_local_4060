# Customer Eval Schema

## Purpose

This JSONL schema is for customer-specific benchmark cases.
Each line is one JSON object.

Use it when you want to:
- compare SemOp against a customer baseline
- measure relation recovery and answer coverage on real SOP cases
- track whether clarification is needed before answering

## Fields

- `id`
  - stable case identifier
- `domain`
  - for example `warehouse_onboarding` or `warehouse_exception`
- `scenario`
  - for example `onboarding`, `quality_gate`, `exception_response`
- `query`
  - the user question
- `context`
  - the relevant SOP or manual excerpt
- `expected_relations`
  - relation types that should be recovered, such as `REQUIRES`, `BLOCKED_BY`, `ALTERNATIVE`
- `expected_answer_terms`
  - terms that should appear in a good answer
- `forbidden_phrases`
  - phrases that must not appear as recommended advice
- `expected_clarification`
  - whether the assistant should ideally ask for clarification before answering
- `metadata`
  - arbitrary customer metadata such as owner, severity, document id, or business unit

## Example

```json
{
  "id": "blocked_aisle_stop",
  "domain": "warehouse_exception",
  "scenario": "exception_response",
  "query": "지게차로 팔레트를 랙에 올리려는데 통로가 막혀 있고 아직 승인도 안 났습니다. 어떻게 해야 하나요?",
  "context": "예외 대응 SOP ...",
  "expected_relations": ["REQUIRES", "BLOCKED_BY", "ALTERNATIVE"],
  "expected_answer_terms": ["관리자 승인", "안전 확인", "스테이징 구역"],
  "forbidden_phrases": ["그냥 이동", "바로 랙으로 이동"],
  "expected_clarification": false,
  "metadata": {"severity": "high", "owner": "warehouse_ops"}
}
```

## How To Create It

### Option A: Start from a template

```bash
python tools/ops/build_customer_eval_template.py --output examples\customer_ops_eval_template.jsonl
```

### Option B: Generate stubs from customer documents

Put customer `.md` or `.txt` files in a folder such as `data\customer_docs`, then run:

```bash
python tools/ops/build_customer_eval_from_docs.py --inputs data\customer_docs --output data\customer_eval.jsonl --max-cases 50
```

Then manually review the generated file.

## How To Benchmark

### Lexical baseline

```bash
python tools/eval/compare_ops_baseline.py --input data\customer_eval.jsonl --baseline lexical_rag
```

### Customer-config baseline

```bash
python tools/eval/compare_ops_baseline.py --input data\customer_eval.jsonl --baseline configurable_keyword --baseline-config data\customer_baseline_config.json
```

## What To Review Manually

Always review these fields before trusting the benchmark:
- `expected_relations`
- `expected_answer_terms`
- `forbidden_phrases`
- `expected_clarification`

This repository can help generate stubs, but the final benchmark still depends on domain review.

