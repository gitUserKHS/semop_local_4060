# Data Collection API Research

## Goal

This project needs two complementary data channels for VLSO:
- low-friction image search APIs for quickly collecting diverse seed images
- larger labeled datasets for detector, segmentation, and geometry-topology alignment

The current implementation target is not mass scraping. It is a controlled, license-aware acquisition loop that produces:
- candidate image URLs and metadata
- local download manifests
- normalized JSONL rows for manual review or later ingestion
- support for few-shot concept memory and prototype training

## Best Fit For This Repository

### Tier 1: Best starter sources

1. `Wikimedia Commons` via the MediaWiki Action API
   - No API key required.
   - Good fit for openly licensed seed images and metadata-rich curation.
   - Best when you want a small, manually reviewed set for concept bootstrapping.

2. `Openverse`
   - Aggregates openly licensed content across multiple sources.
   - Good fit for broad concept exploration with license and source filters.
   - Best when you want to expand a few-shot set without locking into one provider.

### Tier 2: Useful photo APIs, but key and usage review required

3. `Flickr API`
   - Strong metadata and explicit license fields.
   - Good fit for targeted search and metadata-rich photo recall.
   - Requires an API key and extra care for usage terms.

4. `Pexels API`
   - Fast search, high-quality photos, simple authorization.
   - Good fit for visually clean object photos.
   - No structured geometry labels; treat as image-source only.

5. `Unsplash API`
   - High-quality image search and broad search controls.
   - Good fit for image-search prototyping and qualitative evaluation.
   - Review current usage terms carefully before using it for dataset-building or training workflows.

### Tier 3: Labeled dataset rather than API

6. `Open Images V7`
   - Best fit for segmentation, relationships, and large-scale label coverage.
   - Not a query API in the same sense; better treated as a dataset download source.
   - Strong fit for detector/segmentation alignment and geometric relation supervision.

## Official Notes By Source

### Wikimedia Commons / MediaWiki Action API

Official docs:
- [MediaWiki Action API](https://www.mediawiki.org/wiki/API:Main_page)
- [Commons API](https://commons.wikimedia.org/wiki/Commons:API)
- [API etiquette](https://www.mediawiki.org/wiki/API:Etiquette/en)

Useful official points:
- MediaWiki describes `api.php` endpoints as the standard access path for Wikimedia wikis.
- Commons explicitly points developers to the MediaWiki API on Commons, `Special:ApiHelp`, and `Special:ApiSandbox`.
- MediaWiki etiquette says clients should send an informative `User-Agent` and avoid aggressive parallel request patterns.

Repository recommendation:
- Use this as the default no-key starter source for `bag`, `container`, `opening`, `handle`, `tool`, `strap`, `box`, `door`, `hinge`, and geometry-scene seeds.
- Keep collection conservative and review image pages before local download.

### Openverse

Official docs:
- [Openverse developer docs](https://docs.openverse.org/api/index.html)
- [Openverse JS API client reference](https://docs.openverse.org/packages/js/api_client/index.html)
- [Openverse search algorithm notes](https://docs.openverse.org/_preview/869/reference/search_algorithm.html)
- [Openverse auth and throttling reference](https://docs.openverse.org/_preview/2205/api/reference/authentication_and_throttling.html)

Useful official points:
- Openverse positions itself as a search engine for openly licensed media.
- The API client docs show `/v1/images/` queries with `q`, `license`, and `source` filters.
- The client docs also note that rate-limit backoff must be handled by the caller.

Repository recommendation:
- Use Openverse as the best broad, open-license expansion source after Wikimedia.
- Favor it when you want diversity across providers while still normalizing everything into one local manifest.

### Flickr API

Official docs:
- [API keys](https://www.flickr.com/services/api/misc.api_keys.html)
- [flickr.photos.search](https://www.flickr.com/services/api/flickr.photos.search.html)
- [Developer guide](https://www.flickr.com/services/developer/api)
- [API terms](https://www.flickr.com/help/terms/api)

Useful official points:
- Flickr requires an application key.
- `flickr.photos.search` supports `license` filtering and `extras` such as `license`, `owner_name`, and multiple `url_*` sizes.
- Flickr?s developer guide states a limit of `3600` queries per hour per key.

Repository recommendation:
- Use Flickr only when you want metadata-rich photo search and are ready to manage API keys.
- Good candidate for automated collection after Tier 1 sources are working.

### Unsplash API

Official docs:
- [Unsplash API documentation](https://unsplash.com/documentation)

Useful official points:
- Unsplash uses `Authorization: Client-ID YOUR_ACCESS_KEY` for public authentication.
- `GET /search/photos` supports `query`, `per_page`, `order_by`, `content_filter`, `color`, and `orientation`.
- Unsplash documents demo-mode and production rate limits, with demo-mode capped much lower.

Repository recommendation:
- Use as a quality-oriented optional source, not the first default training source.
- Before large-scale collection, verify the current API and attribution terms for your exact usage.

### Pexels API

Official docs:
- [Pexels API documentation](https://www.pexels.com/api/documentation/)

Useful official points:
- Pexels requires an `Authorization` header with an API key.
- The official docs show `https://api.pexels.com/v1/search?query=...` as the base search endpoint.
- Pexels documents default limits of `200` requests per hour and `20,000` per month, with higher limits available on request.

Repository recommendation:
- Good optional source for clean photographic examples.
- Use for object-centric images, but do not expect relational labels or segmentation metadata.

### Open Images V7

Official docs:
- [Open Images V7](https://storage.googleapis.com/openimages/web/index.html)

Useful official points:
- The public site lists boxes, instance segmentations, relationships, localized narratives, point-level annotations, and image-level labels.

Repository recommendation:
- Best source when this repository needs detector or segmentation supervision rather than just raw image search.
- Treat as a dataset-ingest path, not a lightweight image-search API.

## Recommended Collection Strategy

### Stage 1: Build a small, reviewable seed set

Use:
- `wikimedia_commons`
- `openverse`

Reason:
- open-license friendly
- no or low friction for access
- enough metadata to bootstrap `data/vlso_samples/` and few-shot concept memory

### Stage 2: Add optional keyed photo APIs

Use:
- `flickr`
- `pexels`
- `unsplash`

Reason:
- improve visual diversity and photo quality
- expand concept coverage once the collection workflow is stable

### Stage 3: Add labeled datasets

Use:
- `Open Images`
- later, detector or segmentation outputs from external models

Reason:
- better supervision for `detector_adapters.py`, geometric relations, and object-part-affordance extraction

## What Was Added To This Repository

This repository now includes:
- `tools/vlso/collect_visual_data.py`
- `src/semop/vlso/data_collection.py`
- `examples/vlso_collection_manifest.json`

Current use:
- dry-run planning is the default and safest path
- provider-specific request URLs and auth expectations are normalized into JSONL
- live execution is optional and only makes sense when you have credentials and network access

## Suggested Next Implementation Order

1. Use `wikimedia_commons` and `openverse` first.
2. Review the generated manifest rows manually.
3. Download only the accepted images into `data/vlso_samples/`.
4. Build candidate labels with `tools/vlso/build_visual_concept_candidates.py`.
5. Train prototypes with `tools/vlso/train_visual_concepts.py`.
6. Expand into `flickr`, `pexels`, or `unsplash` only after the review loop is stable.
7. Add `Open Images` or detector outputs when segmentation-aware training becomes the main bottleneck.

## Download Approval Stage

This repository now separates collection into three steps:
- `collect_visual_data.py`: build request plans or fetch normalized records
- `prepare_visual_downloads.py`: whitelist and stage approved downloads
- `build_visual_concept_candidates.py` / `train_visual_concepts.py`: turn approved local images into few-shot concept memory

This keeps public API exploration separate from final local dataset curation.
