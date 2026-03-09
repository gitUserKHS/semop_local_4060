# VLSO Data Collection Guide

## Goal

Build a small but efficient tuning set for general visual concepts and affordances.
The target is not to collect a huge image dump. It is to:
- label a few representative objects
- compress them into prototypes
- reuse those prototypes through logical concept flow
- keep labeling only the most novel or uncertain targets

Good starting labels:
- `BAG_LIKE_CONTAINER`
- `ZIPPER_LIKE_PART`
- `ACCESS_OPENING_CANDIDATE`
- `STRAP_LIKE_PART`
- `HANDLE_LIKE_PART`
- `RECTANGLE_LIKE_OBJECT`

## Recommended Folder Layout

Put images here:
- `data\vlso_samples\`

Keep source notes here:
- `data\vlso_samples\SOURCES.md`

Keep manual labels or templates here:
- `examples\vlso_visual_concepts_template.jsonl`
- `data\vlso_samples\concept_candidates.jsonl`

## Efficient Labeling Strategy

Use this loop instead of labeling everything at once:

1. Collect 5 to 20 diverse images.
2. Build candidate rows from raw images.
3. Label only the most informative targets first.
4. Train compact concept prototypes.
5. Re-run QA and inspect failure cases.
6. Add only the next uncertain or novel targets.

This keeps the dataset small while still improving generalization.

## Minimum Coverage

Try to include:
- 4 to 6 container-like objects
- 3 to 5 opening or zipper-like parts
- 3 to 5 handle or strap-like parts
- 3 to 5 hard negatives such as clothing folds, boxes, shadows, or unrelated blobs

## Fast Local Check

```bash
.\.venv312\Scripts\python.exe vlso_demo.py ^
  --mode deep ^
  --query "What objects are visible here?" ^
  --image-path data\vlso_samples\your_image.jpg ^
  --format json
```

## Automatic Collection Research

If you want to expand beyond manual browser download, use:
- `docs/data_collection_api_research.md`
- `tools/vlso/collect_visual_data.py`
- `tools/vlso/prepare_visual_downloads.py`
- `examples/vlso_collection_manifest.json`

Recommended starter providers:
- `wikimedia_commons`
- `openverse`

These are the best low-friction sources for openly licensed seed images.

## Candidate Label Workflow

### 1. Build candidate rows

```bash
python tools\vlso\build_visual_concept_candidates.py ^
  --inputs data\vlso_samples\backpack_public_domain.jpg data\vlso_samples\military_backpack_cc0.jpg ^
  --output data\vlso_samples\concept_candidates.jsonl ^
  --weights data\vlso_samples\trained_affordance_weights.json
```

Each target row includes:
- `subject_id`
- `bbox`
- `shape_hint`
- `feature_vector`
- `prediction_details`
- `suggested_labels`

### 2. Train compact prototypes from labeled rows

```bash
python tools\vlso\train_visual_concepts.py ^
  --labels examples\vlso_visual_concepts_template.jsonl ^
  --store data\vlso_visual_prototypes.db ^
  --summary-output data\vlso_visual_prototypes_summary.json
```

This compresses labeled examples into prototype records and stores co-occurring label-flow hints.

### 3. Ask which targets to label next

```bash
python tools\vlso\recommend_visual_labels.py ^
  --candidates data\vlso_samples\concept_candidates.jsonl ^
  --concept-store data\vlso_visual_prototypes.db ^
  --weights data\vlso_samples\trained_affordance_weights.json ^
  --limit 10
```

The recommender ranks targets by:
- novelty against existing prototypes
- uncertainty under the current weak classifier

### 4. Use the learned store during QA

```bash
.\.venv312\Scripts\python.exe vlso_demo.py ^
  --mode deep ^
  --query "What objects are visible here?" ^
  --image-path data\vlso_samples\backpack_public_domain.jpg ^
  --concept-store data\vlso_visual_prototypes.db ^
  --affordance-weights data\vlso_samples\trained_affordance_weights.json ^
  --answer-mode structured ^
  --format json
```

## Manual Label Template

Use JSONL rows like this:

```json
{"image_path":"data/vlso_samples/example.jpg","targets":[{"subject_id":"shape_1","positive_labels":["BAG_LIKE_CONTAINER"],"negative_labels":[],"notes":"main object"},{"subject_id":"shape_2","positive_labels":["ZIPPER_LIKE_PART"],"negative_labels":[],"notes":"thin opening strip"}]}
```

## Practical Note

If public hosts rate-limit downloads, use:
- manual browser download
- your own phone images
- a small number of diverse scenes instead of many near-duplicates

For this pipeline, diversity matters more than raw image count.

## Labeled Dataset Ingest

If raw search images are not enough, convert `Open Images` box CSV files into VLSO detector JSONL with `tools/vlso/ingest_open_images_annotations.py`.
This is the fastest path from public labeled data into `DetectorOutputAdapter`.
