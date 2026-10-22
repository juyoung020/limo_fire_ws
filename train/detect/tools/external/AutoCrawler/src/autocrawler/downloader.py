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

import base64
import logging
import shutil
from pathlib import Path

import requests

from .sites import Site

logger = logging.getLogger(__name__)

_MAGIC_BYTES: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "jpg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
)


def get_extension_from_link(link: str, default: str = "jpg") -> str:
    ext = link.rsplit(".", 1)[-1].lower() if "." in link else default
    if ext in ("jpg", "jpeg"):
        return "jpg"
    if ext in ("gif", "png"):
        return ext
    return default


def validate_image(path: Path) -> str | None:
    """Detect the real image format from magic bytes.

    `imghdr` was removed in Python 3.13, so this replaces it with a minimal
    sniffer covering the formats this crawler actually downloads.
    """
    header = path.read_bytes()[:12]

    for magic, ext in _MAGIC_BYTES:
        if header.startswith(magic):
            return ext
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "webp"
    return None  # not a valid/supported image


def base64_to_object(src: str) -> bytes:
    _, encoded = src.split(",", 1)
    return base64.decodebytes(encoded.encode("utf-8"))


def download_images(keyword: str, links: list[str], site: Site, download_path: Path, max_count: int = 0) -> None:
    keyword_dir = download_path / keyword.replace('"', "")
    keyword_dir.mkdir(parents=True, exist_ok=True)

    max_count = max_count or len(links)
    success_count = 0

    for index, link in enumerate(links):
        if success_count >= max_count:
            break

        no_ext_path = keyword_dir / f"{site.value}_{index:04d}"
        path = None

        try:
            logger.info("Downloading %s from %s: %d / %d", keyword, site.value, success_count + 1, max_count)

            if link.startswith("data:image/jpeg;base64"):
                path = no_ext_path.with_suffix(".jpg")
                path.write_bytes(base64_to_object(link))
                ext = "jpg"
            elif link.startswith("data:image/png;base64"):
                path = no_ext_path.with_suffix(".png")
                path.write_bytes(base64_to_object(link))
                ext = "png"
            else:
                ext = get_extension_from_link(link)
                path = no_ext_path.with_suffix(f".{ext}")
                response = requests.get(link, stream=True, timeout=10)
                with path.open("wb") as file:
                    shutil.copyfileobj(response.raw, file)
                del response

            success_count += 1

            real_ext = validate_image(path)
            if real_ext is None:
                logger.warning("Unreadable file - %s", link)
                path.unlink()
                success_count -= 1
            elif real_ext != ext:
                renamed = path.with_suffix(f".{real_ext}")
                path.rename(renamed)
                logger.info("Renamed extension %s -> %s", ext, real_ext)

        except KeyboardInterrupt:
            break
        except Exception as exc:  # noqa: BLE001 - one bad link shouldn't stop the batch
            logger.warning("Download failed - %s", exc)
            continue
