from __future__ import annotations

from .artifact_promotion import build_argument_parser, main

__all__ = ["build_argument_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
