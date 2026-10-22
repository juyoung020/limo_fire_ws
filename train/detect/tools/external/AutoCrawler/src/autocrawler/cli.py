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

import argparse
import logging
from pathlib import Path

from .config import CrawlConfig
from .crawler import run

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Google & Naver multiprocess image crawler")
    parser.add_argument(
        "--skip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip a keyword if its download directory already exists. Needed when re-downloading.",
    )
    parser.add_argument("--threads", type=int, default=4, help="Number of worker processes to download with.")
    parser.add_argument(
        "--google", action=argparse.BooleanOptionalAction, default=True, help="Download from google.com."
    )
    parser.add_argument(
        "--naver", action=argparse.BooleanOptionalAction, default=True, help="Download from naver.com."
    )
    parser.add_argument(
        "--full",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Download full resolution images instead of thumbnails (slow, and its selectors are unverified).",
    )
    parser.add_argument(
        "--face", action=argparse.BooleanOptionalAction, default=False, help="Face search mode."
    )
    parser.add_argument(
        "--no_gui",
        choices=["auto", "true", "false"],
        default="auto",
        help='Headless mode. "auto": headless when --full is set. Useful on headless servers, '
        "but unstable in thumbnail mode.",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Maximum count of images to download per site. (0: infinite)"
    )
    parser.add_argument(
        "--proxy-list",
        type=str,
        default="",
        help='Comma separated proxy list, e.g. "socks://127.0.0.1:1080,http://127.0.0.1:1081". '
        "Every task randomly chooses one from the list.",
    )
    parser.add_argument("--download-path", type=Path, default=Path("download"), help="Download folder path.")
    parser.add_argument(
        "--keywords-file", type=Path, default=Path("keywords.txt"), help="Path to the search keywords file."
    )
    return parser.parse_args(argv)


def _resolve_no_gui(value: str, full_resolution: bool) -> bool:
    if value == "auto":
        return full_resolution
    return value == "true"


def _build_config(args: argparse.Namespace) -> CrawlConfig:
    proxy_list = [proxy for proxy in args.proxy_list.split(",") if proxy]

    return CrawlConfig(
        keywords_file=args.keywords_file,
        download_path=args.download_path,
        skip_already_exist=args.skip,
        n_processes=args.threads,
        do_google=args.google,
        do_naver=args.naver,
        full_resolution=args.full,
        face=args.face,
        no_gui=_resolve_no_gui(args.no_gui, args.full),
        limit=args.limit,
        proxy_list=proxy_list,
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = _parse_args(argv)
    config = _build_config(args)

    logger.info("Options - %s", config)
    run(config)


if __name__ == "__main__":
    main()
