from __future__ import annotations

from dataclasses import dataclass, field
import re

from .structures import SymbolicResult


FINANCE_TERMS = {
    "revenue",
    "operating",
    "income",
    "cash",
    "flow",
    "assets",
    "liabilities",
    "margin",
    "debt",
    "profit",
    "expense",
    "guidance",
    "quarter",
    "fiscal",
    "statement",
    "balance",
}

QUESTION_TOKENS = {"what", "which", "why", "how", "who", "when", "어떻게", "무엇", "왜", "어느", "필요", "맞을까요"}


@dataclass
class DocumentBlock:
    text: str
    kind: str
    heading: str = ""
    cells: list[str] = field(default_factory=list)


@dataclass
class EvidenceCandidate:
    block: DocumentBlock
    score: float


class DocumentEvidenceReasoner:
    def solve(self, query: str) -> SymbolicResult | None:
        blocks = self._build_blocks(query)
        if len(blocks) < 2:
            return None
        question = self._pick_question_block(blocks)
        if question is None:
            return None
        question_tokens = self._content_tokens(question.text)
        if len(question_tokens) < 2:
            return None

        candidates = [block for block in blocks if block.text != question.text]
        scored = [EvidenceCandidate(block=block, score=self._score_block(question.text, question_tokens, block)) for block in candidates]
        scored = [candidate for candidate in scored if candidate.score > 0]
        if not scored:
            return None

        scored.sort(key=lambda item: (-item.score, len(item.block.text)))
        top_blocks = [candidate.block for candidate in scored[:2]]
        top_text = " ".join(block.text for block in top_blocks)
        return SymbolicResult(
            domain="document_grounding",
            answer=f"Document evidence points to: {top_text}",
            evidence=[block.text for block in top_blocks],
            equations=[],
            confidence=min(0.88, round(0.58 + 0.04 * scored[0].score, 2)),
            source="symbolic_document",
        )

    def _build_blocks(self, text: str) -> list[DocumentBlock]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) <= 1:
            lines = re.split(r"(?<=[.!?])\s+|;\s+", text)
            lines = [line.strip() for line in lines if line.strip()]

        blocks: list[DocumentBlock] = []
        current_heading = ""
        pending_table_headers: list[str] = []

        for line in lines:
            if self._is_heading(line):
                current_heading = line.rstrip(":")
                continue

            if self._is_table_row(line):
                cells = self._split_table_cells(line)
                if not pending_table_headers and self._looks_like_table_header(cells):
                    pending_table_headers = cells
                    continue
                text_value = self._format_table_block(current_heading, pending_table_headers, cells)
                blocks.append(DocumentBlock(text=text_value, kind="table_row", heading=current_heading, cells=cells))
                continue

            pending_table_headers = []
            for sentence in self._split_sentences(line):
                if len(sentence.split()) < 2:
                    continue
                if current_heading:
                    sentence = f"{current_heading}: {sentence}"
                blocks.append(DocumentBlock(text=sentence, kind="sentence", heading=current_heading))

        if not blocks:
            return []
        return blocks

    def _score_block(self, question: str, question_tokens: set[str], block: DocumentBlock) -> float:
        block_tokens = self._content_tokens(block.text)
        overlap = len(question_tokens & block_tokens)
        if overlap == 0:
            return 0.0
        numeric_overlap = self._numeric_overlap(question, block.text)
        finance_bonus = self._finance_bonus(question_tokens, block_tokens)
        heading_bonus = self._heading_bonus(question_tokens, block.heading)
        table_bonus = self._table_bonus(question_tokens, block)
        return overlap + numeric_overlap + finance_bonus + heading_bonus + table_bonus

    def _pick_question_block(self, blocks: list[DocumentBlock]) -> DocumentBlock | None:
        for block in reversed(blocks):
            lowered = block.text.lower()
            if "?" in block.text or any(token in lowered for token in QUESTION_TOKENS):
                return block
        return blocks[-1] if blocks else None

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        pieces = re.split(r"(?<=[.!?])\s+", text)
        return [piece.strip() for piece in pieces if piece.strip()]

    @staticmethod
    def _is_heading(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if stripped.endswith(":") and len(stripped.split()) <= 6:
            return True
        if stripped.isupper() and len(stripped.split()) <= 6:
            return True
        if len(stripped.split()) <= 4 and any(term in stripped.lower() for term in FINANCE_TERMS):
            return True
        return False

    @staticmethod
    def _is_table_row(line: str) -> bool:
        if "|" in line or "\t" in line:
            return True
        return len(re.split(r"\s{2,}", line.strip())) >= 3

    @staticmethod
    def _split_table_cells(line: str) -> list[str]:
        if "|" in line:
            cells = [cell.strip() for cell in line.split("|") if cell.strip()]
            if cells:
                return cells
        if "\t" in line:
            cells = [cell.strip() for cell in line.split("\t") if cell.strip()]
            if cells:
                return cells
        return [cell.strip() for cell in re.split(r"\s{2,}", line.strip()) if cell.strip()]

    def _looks_like_table_header(self, cells: list[str]) -> bool:
        normalized = [cell.strip().lower() for cell in cells]
        header_leads = {"metric", "metrics", "item", "items", "category", "categories", "year", "years", "quarter", "quarters", "period"}
        if normalized and normalized[0] in header_leads:
            return True
        if len(cells) >= 2 and not re.search(r"\d", cells[0]) and all(re.fullmatch(r"(?:fy)?\d{4}|q[1-4]|[1-4]q", cell.strip().lower()) for cell in cells[1:]):
            return True
        return all(not re.search(r"\d", cell) for cell in cells)

    @staticmethod
    def _format_table_block(heading: str, headers: list[str], cells: list[str]) -> str:
        if headers and len(headers) == len(cells):
            row = ", ".join(f"{header}={cell}" for header, cell in zip(headers, cells))
        else:
            row = " | ".join(cells)
        return f"{heading}: {row}" if heading else row

    @staticmethod
    def _content_tokens(text: str) -> set[str]:
        raw_tokens = re.findall(r"[A-Za-z가-힣0-9]+", text.lower())
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "to", "for", "of", "and", "or", "if", "in", "on", "at", "this", "that",
            "how", "what", "which", "who", "when", "why", "much", "many", "does", "do", "did", "should", "would", "could",
            "은", "는", "이", "가", "을", "를", "에", "의", "도", "과", "와", "으로", "로", "하다", "있는", "합니다", "인가요", "어떻게", "무엇",
        }
        return {token for token in raw_tokens if len(token) > 1 and token not in stopwords}

    @staticmethod
    def _numeric_overlap(question: str, text: str) -> float:
        question_numbers = set(re.findall(r"\d+(?:\.\d+)?", question))
        text_numbers = set(re.findall(r"\d+(?:\.\d+)?", text))
        return 0.5 * len(question_numbers & text_numbers)

    @staticmethod
    def _finance_bonus(question_tokens: set[str], text_tokens: set[str]) -> float:
        shared_finance = (question_tokens & text_tokens) & FINANCE_TERMS
        if shared_finance:
            return 1.5
        if question_tokens & FINANCE_TERMS and text_tokens & FINANCE_TERMS:
            return 1.0
        return 0.0

    def _heading_bonus(self, question_tokens: set[str], heading: str) -> float:
        if not heading:
            return 0.0
        heading_tokens = self._content_tokens(heading)
        if not heading_tokens:
            return 0.0
        return 1.0 if question_tokens & heading_tokens else 0.0

    def _table_bonus(self, question_tokens: set[str], block: DocumentBlock) -> float:
        if block.kind != "table_row":
            return 0.0
        cell_tokens = self._content_tokens(" ".join(block.cells))
        if question_tokens & cell_tokens:
            return 1.0
        return 0.5 if block.heading and (question_tokens & self._content_tokens(block.heading)) else 0.0
