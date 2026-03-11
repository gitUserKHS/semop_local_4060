from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass
class VisualCollectionSource:
    provider: str
    query: str
    limit: int = 20
    license_filters: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    output_tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualCollectionPlanItem:
    provider: str
    query: str
    request_url: str
    headers: Dict[str, str] = field(default_factory=dict)
    auth_env: str | None = None
    notes: List[str] = field(default_factory=list)

    def model_dump(self) -> Dict[str, Any]:
        return {
            'provider': self.provider,
            'query': self.query,
            'request': {
                'url': self.request_url,
                'headers': dict(self.headers),
                'auth_env': self.auth_env,
            },
            'notes': list(self.notes),
        }


@dataclass
class VisualCollectionRecord:
    provider: str
    query: str
    title: str
    page_url: str
    media_url: str
    license: str = ''
    creator: str = ''
    source_id: str = ''
    raw: Dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> Dict[str, Any]:
        return {
            'provider': self.provider,
            'query': self.query,
            'title': self.title,
            'page_url': self.page_url,
            'media_url': self.media_url,
            'license': self.license,
            'creator': self.creator,
            'source_id': self.source_id,
            'raw': dict(self.raw),
        }




@dataclass
class VisualDownloadEntry:
    provider: str
    source_id: str
    media_url: str
    target_path: str
    title: str = ''
    page_url: str = ''
    license: str = ''
    creator: str = ''

    def model_dump(self) -> Dict[str, Any]:
        return {
            'provider': self.provider,
            'source_id': self.source_id,
            'media_url': self.media_url,
            'target_path': self.target_path,
            'title': self.title,
            'page_url': self.page_url,
            'license': self.license,
            'creator': self.creator,
        }


@dataclass
class VisualDownloadSummary:
    records_input: str
    approved_output: str
    manifest_output: str
    approved_count: int
    manifest_count: int
    executed: bool
    downloaded_count: int = 0
    failed_downloads: int = 0

    def model_dump(self) -> Dict[str, Any]:
        return {
            'records_input': self.records_input,
            'approved_output': self.approved_output,
            'manifest_output': self.manifest_output,
            'approved_count': self.approved_count,
            'manifest_count': self.manifest_count,
            'executed': self.executed,
            'downloaded_count': self.downloaded_count,
            'failed_downloads': self.failed_downloads,
        }


@dataclass
class VisualCollectionRunSummary:
    manifest_path: str
    output_path: str
    dry_run: bool
    requests_planned: int
    records_written: int
    providers: List[str]
    failed_requests: int = 0

    def model_dump(self) -> Dict[str, Any]:
        return {
            'manifest_path': self.manifest_path,
            'output_path': self.output_path,
            'dry_run': self.dry_run,
            'requests_planned': self.requests_planned,
            'records_written': self.records_written,
            'providers': list(self.providers),
            'failed_requests': self.failed_requests,
        }


@dataclass
class VisualFamilyBatchSummary:
    workspace: str
    manifest_path: str
    records_path: str
    approved_output: str
    download_manifest_output: str
    download_root: str
    families: Dict[str, int]
    dry_run: bool
    execute_downloads: bool
    manifest_sources: int
    records_written: int
    approved_count: int
    manifest_count: int
    failed_requests: int = 0

    def model_dump(self) -> Dict[str, Any]:
        return {
            'workspace': self.workspace,
            'manifest_path': self.manifest_path,
            'records_path': self.records_path,
            'approved_output': self.approved_output,
            'download_manifest_output': self.download_manifest_output,
            'download_root': self.download_root,
            'families': dict(self.families),
            'dry_run': self.dry_run,
            'execute_downloads': self.execute_downloads,
            'manifest_sources': self.manifest_sources,
            'records_written': self.records_written,
            'approved_count': self.approved_count,
            'manifest_count': self.manifest_count,
            'failed_requests': self.failed_requests,
        }


class VisualDataCollector:
    """Manifest-driven visual data collection planner with optional live fetch support."""

    PROVIDER_DOCS = {
        'wikimedia_commons': 'https://commons.wikimedia.org/wiki/Commons:API',
        'openverse': 'https://api.openverse.org/v1/',
        'flickr': 'https://www.flickr.com/services/api/flickr.photos.search.html',
        'unsplash': 'https://unsplash.com/documentation',
        'pexels': 'https://www.pexels.com/api/documentation/',
    }

    def load_manifest(self, path: str | Path) -> List[VisualCollectionSource]:
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        return [
            VisualCollectionSource(
                provider=str(item['provider']),
                query=str(item['query']),
                limit=int(item.get('limit', 20)),
                license_filters=[str(v) for v in item.get('license_filters', [])],
                categories=[str(v) for v in item.get('categories', [])],
                output_tags=[str(v) for v in item.get('output_tags', [])],
                metadata=dict(item.get('metadata', {})),
            )
            for item in payload.get('sources', [])
        ]

    def build_plan(self, source: VisualCollectionSource) -> VisualCollectionPlanItem:
        provider = source.provider.lower().strip()
        if provider == 'wikimedia_commons':
            params = {
                'action': 'query',
                'format': 'json',
                'generator': 'search',
                'gsrsearch': self._wikimedia_query(source),
                'gsrnamespace': '6',
                'gsrlimit': str(source.limit),
                'prop': 'imageinfo|info',
                'iiprop': 'url|extmetadata',
                'inprop': 'url',
            }
            return VisualCollectionPlanItem(
                provider=provider,
                query=source.query,
                request_url='https://commons.wikimedia.org/w/api.php?' + urlencode(params),
                headers={'User-Agent': self._user_agent()},
                notes=[
                    'No API key required, but MediaWiki API etiquette requires an informative User-Agent and serialized requests.',
                    'Best starter source for openly licensed seed images and rich metadata.',
                ],
            )
        if provider == 'openverse':
            params = {'q': source.query, 'page_size': str(min(source.limit, 50))}
            if source.license_filters:
                params['license'] = ','.join(source.license_filters)
            if source.categories:
                params['category'] = ','.join(source.categories)
            return VisualCollectionPlanItem(
                provider=provider,
                query=source.query,
                request_url='https://api.openverse.org/v1/images/?' + urlencode(params),
                headers={'User-Agent': self._user_agent()},
                notes=[
                    'Useful for broad open-license search across multiple upstream providers.',
                    'Supports filtering by license and source in the public API surface.',
                ],
            )
        if provider == 'flickr':
            api_key = os.environ.get('FLICKR_API_KEY', '<FLICKR_API_KEY>')
            params = {
                'method': 'flickr.photos.search',
                'api_key': api_key,
                'text': source.query,
                'per_page': str(source.limit),
                'format': 'json',
                'nojsoncallback': '1',
                'extras': 'license,owner_name,tags,url_o,url_l,url_c',
            }
            if source.license_filters:
                params['license'] = ','.join(source.license_filters)
            return VisualCollectionPlanItem(
                provider=provider,
                query=source.query,
                request_url='https://www.flickr.com/services/rest/?' + urlencode(params),
                auth_env='FLICKR_API_KEY',
                notes=[
                    'Flickr requires an API key and uses per-key limits.',
                    'Good metadata richness and license extras for targeted collection.',
                ],
            )
        if provider == 'unsplash':
            params = {'query': source.query, 'per_page': str(min(source.limit, 30))}
            headers = {'Authorization': f"Client-ID {os.environ.get('UNSPLASH_ACCESS_KEY', '<UNSPLASH_ACCESS_KEY>')}"}
            return VisualCollectionPlanItem(
                provider=provider,
                query=source.query,
                request_url='https://api.unsplash.com/search/photos?' + urlencode(params),
                headers=headers,
                auth_env='UNSPLASH_ACCESS_KEY',
                notes=[
                    'Unsplash requires an access key and enforces demo/production rate limits.',
                    'Use only after checking current API terms for dataset or training usage.',
                ],
            )
        if provider == 'pexels':
            params = {'query': source.query, 'per_page': str(min(source.limit, 80))}
            headers = {'Authorization': os.environ.get('PEXELS_API_KEY', '<PEXELS_API_KEY>')}
            return VisualCollectionPlanItem(
                provider=provider,
                query=source.query,
                request_url='https://api.pexels.com/v1/search?' + urlencode(params),
                headers=headers,
                auth_env='PEXELS_API_KEY',
                notes=[
                    'Pexels requires an API key and attribution-oriented usage.',
                    'Good for quickly collecting high-quality photos but provides no segmentation labels.',
                ],
            )
        raise ValueError(f'Unsupported provider: {source.provider}')

    def run_manifest(
        self,
        manifest_path: str | Path,
        output_path: str | Path,
        dry_run: bool = True,
        progress_callback: Callable[[str, Dict[str, Any]], None] | None = None,
    ) -> VisualCollectionRunSummary:
        sources = self.load_manifest(manifest_path)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        providers: List[str] = []
        written = 0
        failed = 0
        with output.open('w', encoding='utf-8') as handle:
            total_sources = len(sources)
            for index, source in enumerate(sources, start=1):
                plan = self.build_plan(source)
                providers.append(plan.provider)
                if progress_callback is not None:
                    progress_callback('plan', {'index': index, 'total': total_sources, 'provider': plan.provider, 'query': plan.query, 'dry_run': dry_run})
                if dry_run:
                    handle.write(json.dumps(plan.model_dump(), ensure_ascii=False) + '\n')
                    written += 1
                    continue
                try:
                    records = self.fetch_and_normalize(plan)
                except Exception as exc:
                    failed += 1
                    if progress_callback is not None:
                        progress_callback('request_failed', {'index': index, 'total': total_sources, 'provider': plan.provider, 'query': plan.query, 'error': str(exc)})
                    continue
                if progress_callback is not None:
                    progress_callback('request_succeeded', {'index': index, 'total': total_sources, 'provider': plan.provider, 'query': plan.query, 'records': len(records)})
                for record in records:
                    handle.write(json.dumps(record.model_dump(), ensure_ascii=False) + '\n')
                    written += 1
        return VisualCollectionRunSummary(
            manifest_path=str(manifest_path),
            output_path=str(output_path),
            dry_run=dry_run,
            requests_planned=len(sources),
            records_written=written,
            providers=sorted(set(providers)),
            failed_requests=failed,
        )

    def fetch_and_normalize(self, plan: VisualCollectionPlanItem) -> List[VisualCollectionRecord]:
        request = Request(plan.request_url, headers=plan.headers)
        payload = json.loads(self._read_bytes_with_retries(request, timeout=30).decode('utf-8'))
        normalizer = getattr(self, f'_normalize_{plan.provider}', None)
        if normalizer is None:
            return [VisualCollectionRecord(provider=plan.provider, query=plan.query, title='', page_url='', media_url='', raw=payload)]
        return normalizer(plan.query, payload)

    def run_family_batch(
        self,
        family_targets: Dict[str, int],
        workspace: str | Path,
        execute_collect: bool = False,
        execute_downloads: bool = False,
        allow_providers: List[str] | None = None,
        allow_licenses: List[str] | None = None,
        accept_all: bool = True,
        progress_callback: Callable[[str, Dict[str, Any]], None] | None = None,
    ) -> VisualFamilyBatchSummary:
        workspace_path = Path(workspace)
        workspace_path.mkdir(parents=True, exist_ok=True)
        normalized_targets = {str(key).strip().lower(): max(1, int(value)) for key, value in family_targets.items() if str(key).strip()}
        families = list(normalized_targets)
        per_family_limits: dict[str, int] = {}
        for family in families:
            preset_count = max(1, len(FAMILY_QUERY_PRESETS.get(family, [])))
            per_family_limits[family] = max(4, math.ceil(normalized_targets[family] / preset_count))
        max_limit = max(per_family_limits.values(), default=8)
        manifest_payload = build_object_family_manifest(families, limit_per_source=max_limit)
        for row in manifest_payload.get('sources', []):
            if not isinstance(row, dict):
                continue
            family = str((row.get('metadata') or {}).get('family', '')).lower()
            if family in per_family_limits:
                row['limit'] = per_family_limits[family]
        manifest_path = workspace_path / 'family_manifest.json'
        records_path = workspace_path / 'family_records.jsonl'
        approved_path = workspace_path / 'family_approved.jsonl'
        download_manifest_path = workspace_path / 'family_download_manifest.jsonl'
        download_root = workspace_path / 'downloads'
        manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding='utf-8')
        collection_summary = self.run_manifest(manifest_path, records_path, dry_run=not execute_collect, progress_callback=progress_callback)
        approved_count = 0
        manifest_count = 0
        if execute_collect and records_path.exists():
            download_summary = self.prepare_downloads(
                records_path=records_path,
                approved_output=approved_path,
                manifest_output=download_manifest_path,
                download_root=download_root,
                allow_providers=allow_providers,
                allow_licenses=allow_licenses,
                accept_all=accept_all,
                execute=execute_downloads,
                progress_callback=progress_callback,
            )
            approved_count = download_summary.approved_count
            manifest_count = download_summary.manifest_count
        return VisualFamilyBatchSummary(
            workspace=str(workspace_path),
            manifest_path=str(manifest_path),
            records_path=str(records_path),
            approved_output=str(approved_path),
            download_manifest_output=str(download_manifest_path),
            download_root=str(download_root),
            families=normalized_targets,
            dry_run=not execute_collect,
            execute_downloads=execute_downloads,
            manifest_sources=len(manifest_payload.get('sources', [])),
            records_written=collection_summary.records_written,
            approved_count=approved_count,
            manifest_count=manifest_count,
            failed_requests=collection_summary.failed_requests,
        )

    def _normalize_wikimedia_commons(self, query: str, payload: Dict[str, Any]) -> List[VisualCollectionRecord]:
        pages = list(payload.get('query', {}).get('pages', {}).values())
        rows: List[VisualCollectionRecord] = []
        for page in pages:
            imageinfo = next(iter(page.get('imageinfo', []) or []), {})
            metadata = imageinfo.get('extmetadata', {})
            rows.append(VisualCollectionRecord(
                provider='wikimedia_commons',
                query=query,
                title=str(page.get('title', '')),
                page_url=str(page.get('fullurl', '')),
                media_url=str(imageinfo.get('url', '')),
                license=str(metadata.get('LicenseShortName', {}).get('value', '')),
                creator=str(metadata.get('Artist', {}).get('value', '')),
                source_id=str(page.get('pageid', '')),
                raw=page,
            ))
        return rows

    def _normalize_openverse(self, query: str, payload: Dict[str, Any]) -> List[VisualCollectionRecord]:
        return [VisualCollectionRecord(
            provider='openverse',
            query=query,
            title=str(item.get('title', '')),
            page_url=str(item.get('foreign_landing_url', '')),
            media_url=str(item.get('url', '')),
            license=str(item.get('license', '')),
            creator=str(item.get('creator', '')),
            source_id=str(item.get('id', '')),
            raw=item,
        ) for item in payload.get('results', [])]

    def _normalize_unsplash(self, query: str, payload: Dict[str, Any]) -> List[VisualCollectionRecord]:
        return [VisualCollectionRecord(
            provider='unsplash',
            query=query,
            title=str(item.get('description') or item.get('alt_description') or ''),
            page_url=str(item.get('links', {}).get('html', '')),
            media_url=str(item.get('urls', {}).get('regular', '')),
            license='unsplash',
            creator=str(item.get('user', {}).get('name', '')),
            source_id=str(item.get('id', '')),
            raw=item,
        ) for item in payload.get('results', [])]

    def _normalize_pexels(self, query: str, payload: Dict[str, Any]) -> List[VisualCollectionRecord]:
        return [VisualCollectionRecord(
            provider='pexels',
            query=query,
            title=str(item.get('alt', '')),
            page_url=str(item.get('url', '')),
            media_url=str(item.get('src', {}).get('large', '')),
            license='pexels',
            creator=str(item.get('photographer', '')),
            source_id=str(item.get('id', '')),
            raw=item,
        ) for item in payload.get('photos', [])]

    def _normalize_flickr(self, query: str, payload: Dict[str, Any]) -> List[VisualCollectionRecord]:
        rows: List[VisualCollectionRecord] = []
        for item in payload.get('photos', {}).get('photo', []):
            media_url = str(item.get('url_o') or item.get('url_l') or item.get('url_c') or '')
            rows.append(VisualCollectionRecord(
                provider='flickr',
                query=query,
                title=str(item.get('title', '')),
                page_url='',
                media_url=media_url,
                license=str(item.get('license', '')),
                creator=str(item.get('ownername', '')),
                source_id=str(item.get('id', '')),
                raw=item,
            ))
        return rows



    def load_records(self, path: str | Path) -> List[VisualCollectionRecord]:
        rows: List[VisualCollectionRecord] = []
        with Path(path).open('r', encoding='utf-8-sig') as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if 'request' in row:
                    continue
                rows.append(
                    VisualCollectionRecord(
                        provider=str(row.get('provider', '')),
                        query=str(row.get('query', '')),
                        title=str(row.get('title', '')),
                        page_url=str(row.get('page_url', '')),
                        media_url=str(row.get('media_url', '')),
                        license=str(row.get('license', '')),
                        creator=str(row.get('creator', '')),
                        source_id=str(row.get('source_id', '')),
                        raw=dict(row.get('raw', {})),
                    )
                )
        return rows

    def prepare_downloads(
        self,
        records_path: str | Path,
        approved_output: str | Path,
        manifest_output: str | Path,
        download_root: str | Path,
        allow_providers: Iterable[str] | None = None,
        allow_licenses: Iterable[str] | None = None,
        approved_ids: Iterable[str] | None = None,
        accept_all: bool = False,
        execute: bool = False,
        progress_callback: Callable[[str, Dict[str, Any]], None] | None = None,
    ) -> VisualDownloadSummary:
        records = self.load_records(records_path)
        approved = self._filter_records(
            records,
            allow_providers={item.lower() for item in allow_providers} if allow_providers else None,
            allow_licenses={item.lower() for item in allow_licenses} if allow_licenses else None,
            approved_ids={str(item) for item in approved_ids} if approved_ids else None,
            accept_all=accept_all,
        )
        approved_path = Path(approved_output)
        approved_path.parent.mkdir(parents=True, exist_ok=True)
        with approved_path.open('w', encoding='utf-8') as handle:
            for record in approved:
                handle.write(json.dumps(record.model_dump(), ensure_ascii=False) + '\n')
        manifest = self.build_download_manifest(approved, download_root)
        manifest_path = Path(manifest_output)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open('w', encoding='utf-8') as handle:
            for item in manifest:
                handle.write(json.dumps(item.model_dump(), ensure_ascii=False) + '\n')
        downloaded_count = 0
        failed_downloads = 0
        if execute:
            downloaded_count, failed_downloads = self.execute_download_manifest(manifest, progress_callback=progress_callback)
        return VisualDownloadSummary(
            records_input=str(records_path),
            approved_output=str(approved_output),
            manifest_output=str(manifest_output),
            approved_count=len(approved),
            manifest_count=len(manifest),
            executed=execute,
            downloaded_count=downloaded_count,
            failed_downloads=failed_downloads,
        )

    def build_download_manifest(self, records: Iterable[VisualCollectionRecord], download_root: str | Path) -> List[VisualDownloadEntry]:
        root = Path(download_root)
        entries: List[VisualDownloadEntry] = []
        for record in records:
            ext = self._guess_extension(record.media_url)
            safe_name = self._safe_name(record.source_id or record.title or 'item')
            target_path = root / record.provider / f'{safe_name}{ext}'
            entries.append(
                VisualDownloadEntry(
                    provider=record.provider,
                    source_id=record.source_id,
                    media_url=record.media_url,
                    target_path=str(target_path),
                    title=record.title,
                    page_url=record.page_url,
                    license=record.license,
                    creator=record.creator,
                )
            )
        return entries

    def execute_download_manifest(
        self,
        entries: Iterable[VisualDownloadEntry],
        progress_callback: Callable[[str, Dict[str, Any]], None] | None = None,
    ) -> tuple[int, int]:
        downloaded_count = 0
        failed_downloads = 0
        items = list(entries)
        total = len(items)
        for index, entry in enumerate(items, start=1):
            if not entry.media_url:
                failed_downloads += 1
                continue
            target = Path(entry.target_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            request = Request(entry.media_url, headers={'User-Agent': self._user_agent()})
            if progress_callback is not None:
                progress_callback('download_start', {'index': index, 'total': total, 'provider': entry.provider, 'title': entry.title, 'target_path': entry.target_path})
            try:
                payload = self._read_bytes_with_retries(request, timeout=60)
            except Exception as exc:
                failed_downloads += 1
                if progress_callback is not None:
                    progress_callback('download_failed', {'index': index, 'total': total, 'provider': entry.provider, 'title': entry.title, 'error': str(exc)})
                continue
            with target.open('wb') as handle:
                handle.write(payload)
            downloaded_count += 1
            if progress_callback is not None:
                progress_callback('download_succeeded', {'index': index, 'total': total, 'provider': entry.provider, 'title': entry.title, 'target_path': entry.target_path})
        return downloaded_count, failed_downloads

    def _filter_records(
        self,
        records: Iterable[VisualCollectionRecord],
        allow_providers: set[str] | None = None,
        allow_licenses: set[str] | None = None,
        approved_ids: set[str] | None = None,
        accept_all: bool = False,
    ) -> List[VisualCollectionRecord]:
        approved: List[VisualCollectionRecord] = []
        for record in records:
            if not record.media_url:
                continue
            if approved_ids is not None:
                if record.source_id in approved_ids or record.media_url in approved_ids:
                    approved.append(record)
                continue
            if accept_all:
                approved.append(record)
                continue
            if allow_providers is not None and record.provider.lower() not in allow_providers:
                continue
            if allow_licenses is not None:
                license_value = record.license.lower().strip()
                if license_value and all(token not in license_value for token in allow_licenses):
                    continue
            approved.append(record)
        return approved

    @staticmethod
    def _safe_name(value: str) -> str:
        cleaned = ''.join(ch if ch.isalnum() or ch in {'-', '_'} else '_' for ch in value.strip())
        return cleaned.strip('_') or 'item'

    @staticmethod
    def _guess_extension(url: str) -> str:
        lowered = url.lower()
        for ext in ('.png', '.webp', '.jpeg', '.jpg', '.gif'):
            if ext in lowered:
                return ext
        return '.jpg'

    @staticmethod
    def _request_delay_seconds() -> float:
        try:
            return max(0.0, float(os.environ.get('SEMOP_VLSO_REQUEST_DELAY_SECONDS', '0.6')))
        except ValueError:
            return 0.6

    @staticmethod
    def _max_retries() -> int:
        try:
            return max(1, int(os.environ.get('SEMOP_VLSO_MAX_RETRIES', '4')))
        except ValueError:
            return 4

    def _retry_delay_seconds(self, attempt_index: int, error: Exception | None = None) -> float:
        if isinstance(error, HTTPError):
            retry_after = error.headers.get('Retry-After') if error.headers else None
            if retry_after:
                try:
                    return max(float(retry_after), self._request_delay_seconds())
                except ValueError:
                    pass
        base = max(0.5, self._request_delay_seconds())
        return base * (2 ** attempt_index)

    def _read_bytes_with_retries(self, request: Request, timeout: int) -> bytes:
        attempts = self._max_retries()
        last_error: Exception | None = None
        for attempt in range(attempts):
            if attempt > 0:
                time.sleep(self._retry_delay_seconds(attempt - 1, last_error))
            try:
                with urlopen(request, timeout=timeout) as response:
                    payload = response.read()
                delay = self._request_delay_seconds()
                if delay > 0:
                    time.sleep(delay)
                return payload
            except HTTPError as exc:
                last_error = exc
                if exc.code not in {429, 500, 502, 503, 504} or attempt >= attempts - 1:
                    raise
            except URLError as exc:
                last_error = exc
                if attempt >= attempts - 1:
                    raise
        if last_error is not None:
            raise last_error
        raise RuntimeError('visual data request failed without an exception')


    @staticmethod
    def _wikimedia_query(source: VisualCollectionSource) -> str:
        query = source.query
        if source.categories:
            category_terms = ' '.join(f'filetype:bitmap incategory:"{category}"' for category in source.categories)
            query = f'{query} {category_terms}'.strip()
        return query

    @staticmethod
    def _user_agent() -> str:
        return 'SemOpLocal4060/1.0 (local research prototype; contact: local-user)'


def build_geometry_seed_manifest() -> Dict[str, Any]:
    return {
        'sources': [
            {
                'provider': 'wikimedia_commons',
                'query': 'triangle diagram',
                'limit': 12,
                'categories': ['Triangles'],
                'output_tags': ['geometry', 'triangle', 'diagram'],
                'metadata': {'purpose': 'vlso_geometry_seed'},
            },
            {
                'provider': 'wikimedia_commons',
                'query': 'rectangle diagram parallel lines',
                'limit': 12,
                'categories': ['Rectangles'],
                'output_tags': ['geometry', 'rectangle', 'parallel'],
                'metadata': {'purpose': 'vlso_geometry_seed'},
            },
            {
                'provider': 'openverse',
                'query': 'open drawer handle',
                'limit': 10,
                'license_filters': ['by', 'cc0'],
                'output_tags': ['drawer', 'handle', 'opening'],
                'metadata': {'purpose': 'vlso_access_seed'},
            },
            {
                'provider': 'openverse',
                'query': 'door hinge open',
                'limit': 10,
                'license_filters': ['by', 'cc0'],
                'output_tags': ['door', 'hinge', 'opening'],
                'metadata': {'purpose': 'vlso_access_seed'},
            },
            {
                'provider': 'openverse',
                'query': 'bottle cap close up',
                'limit': 10,
                'license_filters': ['by', 'cc0'],
                'output_tags': ['bottle', 'cap', 'opening'],
                'metadata': {'purpose': 'vlso_access_seed'},
            },
            {
                'provider': 'openverse',
                'query': 'tool box handle',
                'limit': 10,
                'license_filters': ['by', 'cc0'],
                'output_tags': ['tool', 'box', 'handle'],
                'metadata': {'purpose': 'vlso_tool_seed'},
            },
        ]
    }


FAMILY_QUERY_PRESETS: Dict[str, List[Dict[str, Any]]] = {
    'bag': [
        {'provider': 'wikimedia_commons', 'query': 'backpack zipper', 'categories': ['Backpacks'], 'output_tags': ['bag', 'container', 'opening']},
        {'provider': 'openverse', 'query': 'bag handle opening', 'license_filters': ['by', 'cc0'], 'output_tags': ['bag', 'handle', 'opening']},
        {'provider': 'openverse', 'query': 'messenger bag strap', 'license_filters': ['by', 'cc0'], 'output_tags': ['bag', 'strap', 'carry']},
    ],
    'box': [
        {'provider': 'wikimedia_commons', 'query': 'storage box lid', 'categories': ['Boxes'], 'output_tags': ['box', 'lid', 'container']},
        {'provider': 'openverse', 'query': 'tool box open handle', 'license_filters': ['by', 'cc0'], 'output_tags': ['box', 'handle', 'opening']},
        {'provider': 'openverse', 'query': 'plastic storage bin lid', 'license_filters': ['by', 'cc0'], 'output_tags': ['box', 'bin', 'lid']},
    ],
    'drawer': [
        {'provider': 'openverse', 'query': 'open drawer handle', 'license_filters': ['by', 'cc0'], 'output_tags': ['drawer', 'handle', 'opening']},
        {'provider': 'wikimedia_commons', 'query': 'desk drawer open', 'categories': ['Drawers'], 'output_tags': ['drawer', 'container', 'opening']},
        {'provider': 'openverse', 'query': 'cabinet drawer pull', 'license_filters': ['by', 'cc0'], 'output_tags': ['drawer', 'pull', 'handle']},
    ],
    'door': [
        {'provider': 'openverse', 'query': 'door hinge open', 'license_filters': ['by', 'cc0'], 'output_tags': ['door', 'hinge', 'opening']},
        {'provider': 'wikimedia_commons', 'query': 'door handle open', 'categories': ['Doors'], 'output_tags': ['door', 'handle', 'opening']},
        {'provider': 'openverse', 'query': 'cabinet door knob', 'license_filters': ['by', 'cc0'], 'output_tags': ['door', 'knob', 'handle']},
    ],
    'bottle': [
        {'provider': 'openverse', 'query': 'bottle cap close up', 'license_filters': ['by', 'cc0'], 'output_tags': ['bottle', 'cap', 'opening']},
        {'provider': 'wikimedia_commons', 'query': 'water bottle cap', 'categories': ['Bottles'], 'output_tags': ['bottle', 'container', 'cap']},
        {'provider': 'openverse', 'query': 'jar lid close up', 'license_filters': ['by', 'cc0'], 'output_tags': ['bottle', 'jar', 'lid']},
    ],
    'tool': [
        {'provider': 'openverse', 'query': 'tool handle close up', 'license_filters': ['by', 'cc0'], 'output_tags': ['tool', 'grasp', 'handle']},
        {'provider': 'openverse', 'query': 'tool box handle', 'license_filters': ['by', 'cc0'], 'output_tags': ['tool', 'box', 'handle']},
        {'provider': 'openverse', 'query': 'hammer handle grip', 'license_filters': ['by', 'cc0'], 'output_tags': ['tool', 'grip', 'handle']},
    ],
    'cabinet': [
        {'provider': 'openverse', 'query': 'cabinet door handle', 'license_filters': ['by', 'cc0'], 'output_tags': ['cabinet', 'door', 'handle']},
        {'provider': 'openverse', 'query': 'kitchen cabinet open', 'license_filters': ['by', 'cc0'], 'output_tags': ['cabinet', 'opening', 'storage']},
    ],
    'suitcase': [
        {'provider': 'openverse', 'query': 'suitcase zipper handle', 'license_filters': ['by', 'cc0'], 'output_tags': ['suitcase', 'zipper', 'handle']},
        {'provider': 'wikimedia_commons', 'query': 'luggage case handle', 'categories': ['Suitcases'], 'output_tags': ['suitcase', 'case', 'carry']},
    ],
    'jar': [
        {'provider': 'openverse', 'query': 'glass jar lid close up', 'license_filters': ['by', 'cc0'], 'output_tags': ['jar', 'lid', 'container']},
        {'provider': 'openverse', 'query': 'jar opening cap', 'license_filters': ['by', 'cc0'], 'output_tags': ['jar', 'opening', 'cap']},
    ],
    'bin': [
        {'provider': 'openverse', 'query': 'storage bin lid handle', 'license_filters': ['by', 'cc0'], 'output_tags': ['bin', 'lid', 'handle']},
        {'provider': 'wikimedia_commons', 'query': 'plastic bin container', 'categories': ['Boxes'], 'output_tags': ['bin', 'container', 'storage']},
    ],
    'pouch': [
        {'provider': 'openverse', 'query': 'zipper pouch open', 'license_filters': ['by', 'cc0'], 'output_tags': ['pouch', 'zipper', 'opening']},
        {'provider': 'openverse', 'query': 'small pouch strap', 'license_filters': ['by', 'cc0'], 'output_tags': ['pouch', 'strap', 'carry']},
    ],
}



def build_object_family_manifest(families: List[str], limit_per_source: int = 12) -> Dict[str, Any]:
    normalized = []
    seen = set()
    for family in families:
        key = family.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    sources: List[Dict[str, Any]] = []
    for family in normalized:
        for template in FAMILY_QUERY_PRESETS.get(family, []):
            row = dict(template)
            row['limit'] = int(row.get('limit', limit_per_source))
            row['metadata'] = {'purpose': 'vlso_family_seed', 'family': family}
            row.setdefault('output_tags', [])
            row['output_tags'] = [str(v) for v in row['output_tags']] + [family]
            sources.append(row)
    return {'sources': sources}
