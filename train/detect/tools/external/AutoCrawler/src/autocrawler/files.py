"""
Copyright 2018 YoongiKim

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def get_keywords(keywords_file: Path) -> list[str]:
    """Read, dedupe and sort search keywords, rewriting the file in sorted order."""
    text = keywords_file.read_text(encoding="utf-8-sig")
    lines = [line for line in text.split("\n") if line]
    keywords = sorted(set(lines))

    logger.info("%d keywords found: %s", len(keywords), keywords)

    keywords_file.write_text("".join(f"{keyword}\n" for keyword in keywords), encoding="utf-8")

    return keywords


def all_dirs(path: Path) -> list[Path]:
    return [p for p in path.iterdir() if p.is_dir()]


def all_files(path: Path) -> list[Path]:
    return [p for p in path.rglob("*") if p.is_file()]
