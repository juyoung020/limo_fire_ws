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

from dataclasses import dataclass, field
from pathlib import Path

from .sites import Site


@dataclass(frozen=True)
class CrawlConfig:
    keywords_file: Path = Path("keywords.txt")
    download_path: Path = Path("download")
    chrome_profile_dir: Path = Path("chrome-profile")
    skip_already_exist: bool = True
    n_processes: int = 4
    do_google: bool = True
    do_naver: bool = True
    full_resolution: bool = False
    face: bool = False
    no_gui: bool = False
    limit: int = 0
    proxy_list: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        # Render Path fields as plain strings (e.g. 'keywords.txt') instead of the
        # default dataclass repr's PosixPath('keywords.txt').
        fields = ", ".join(
            f"{name}={str(value)!r}" if isinstance(value, Path) else f"{name}={value!r}"
            for name, value in vars(self).items()
        )
        return f"{type(self).__name__}({fields})"


@dataclass(frozen=True)
class CrawlTask:
    keyword: str
    site: Site
    full_resolution: bool
