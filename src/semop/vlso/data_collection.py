from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
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

    def model_dump(self) -> Dict[str, Any]:
        return {
            'records_input': self.records_input,
            'approved_output': self.approved_output,
            'manifest_output': self.manifest_output,
            'approved_count': self.approved_count,
            'manifest_count': self.manifest_count,
            'executed': self.executed,
        }


@dataclass
class VisualCollectionRunSummary:
    manifest_path: str
    output_path: str
    dry_run: bool
    requests_planned: int
    records_written: int
    providers: List[str]

    def model_dump(self) -> Dict[str, Any]:
        return {
            'manifest_path': self.manifest_path,
            'output_path': self.output_path,
            'dry_run': self.dry_run,
            'requests_planned': self.requests_planned,
            'records_written': self.records_written,
            'providers': list(self.providers),
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

    def run_manifest(self, manifest_path: str | Path, output_path: str | Path, dry_run: bool = True) -> VisualCollectionRunSummary:
        sources = self.load_manifest(manifest_path)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        providers: List[str] = []
        written = 0
        with output.open('w', encoding='utf-8') as handle:
            for source in sources:
                plan = self.build_plan(source)
                providers.append(plan.provider)
                if dry_run:
                    handle.write(json.dumps(plan.model_dump(), ensure_ascii=False) + '\n')
                    written += 1
                    continue
                records = self.fetch_and_normalize(plan)
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
        )

    def fetch_and_normalize(self, plan: VisualCollectionPlanItem) -> List[VisualCollectionRecord]:
        request = Request(plan.request_url, headers=plan.headers)
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode('utf-8'))
        normalizer = getattr(self, f'_normalize_{plan.provider}', None)
        if normalizer is None:
            return [VisualCollectionRecord(provider=plan.provider, query=plan.query, title='', page_url='', media_url='', raw=payload)]
        return normalizer(plan.query, payload)

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
        if execute:
            self.execute_download_manifest(manifest)
        return VisualDownloadSummary(
            records_input=str(records_path),
            approved_output=str(approved_output),
            manifest_output=str(manifest_output),
            approved_count=len(approved),
            manifest_count=len(manifest),
            executed=execute,
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

    def execute_download_manifest(self, entries: Iterable[VisualDownloadEntry]) -> None:
        for entry in entries:
            if not entry.media_url:
                continue
            target = Path(entry.target_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            request = Request(entry.media_url, headers={'User-Agent': self._user_agent()})
            with urlopen(request, timeout=60) as response, target.open('wb') as handle:
                handle.write(response.read())

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
    def _wikimedia_query(source: VisualCollectionSource) -> str:
        query = source.query
        if source.categories:
            category_terms = ' '.join(f'filetype:bitmap incategory:"{category}"' for category in source.categories)
            query = f'{query} {category_terms}'.strip()
        return query

    @staticmethod
    def _user_agent() -> str:
        return 'SemOpLocal4060/1.0 (local research prototype; contact: local-user)'
