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

from autocrawler.files import all_dirs, all_files, get_keywords


def test_get_keywords_dedupes_sorts_and_rewrites(tmp_path: Path):
    keywords_file = tmp_path / "keywords.txt"
    keywords_file.write_text("dog\ncat\ncat\n\n", encoding="utf-8")

    keywords = get_keywords(keywords_file)

    assert keywords == ["cat", "dog"]
    assert keywords_file.read_text(encoding="utf-8") == "cat\ndog\n"


def test_all_dirs_lists_only_directories(tmp_path: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "file.txt").write_text("x")

    dirs = {p.name for p in all_dirs(tmp_path)}
    assert dirs == {"a", "b"}


def test_all_files_lists_files_recursively(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.jpg").write_bytes(b"x")

    files = {p.name for p in all_files(tmp_path)}
    assert files == {"a.jpg", "b.jpg"}
