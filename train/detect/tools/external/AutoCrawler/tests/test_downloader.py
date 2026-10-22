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

from pathlib import Path

from autocrawler.downloader import get_extension_from_link, validate_image

JPEG_HEADER = b"\xff\xd8\xff\xe0" + b"\x00" * 20
PNG_HEADER = b"\x89PNG\r\n\x1a\n" + b"\x00" * 10
GIF_HEADER = b"GIF89a" + b"\x00" * 10
WEBP_HEADER = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 4
INVALID_HEADER = b"not an image" + b"\x00" * 10


def test_get_extension_from_link_normalizes_jpeg():
    assert get_extension_from_link("https://example.com/a.JPEG") == "jpg"


def test_get_extension_from_link_defaults_for_unknown():
    assert get_extension_from_link("https://example.com/a.webp") == "jpg"


def test_get_extension_from_link_defaults_when_no_dot():
    assert get_extension_from_link("https://example.com/noext") == "jpg"


def test_validate_image_detects_jpeg(tmp_path: Path):
    path = tmp_path / "img"
    path.write_bytes(JPEG_HEADER)
    assert validate_image(path) == "jpg"


def test_validate_image_detects_png(tmp_path: Path):
    path = tmp_path / "img"
    path.write_bytes(PNG_HEADER)
    assert validate_image(path) == "png"


def test_validate_image_detects_gif(tmp_path: Path):
    path = tmp_path / "img"
    path.write_bytes(GIF_HEADER)
    assert validate_image(path) == "gif"


def test_validate_image_detects_webp(tmp_path: Path):
    path = tmp_path / "img"
    path.write_bytes(WEBP_HEADER)
    assert validate_image(path) == "webp"


def test_validate_image_rejects_invalid(tmp_path: Path):
    path = tmp_path / "img"
    path.write_bytes(INVALID_HEADER)
    assert validate_image(path) is None
