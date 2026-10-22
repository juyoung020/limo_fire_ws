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

from autocrawler.crawler import find_undersized_dirs


def test_find_undersized_dirs_flags_below_half_average():
    counts = {Path("a"): 100, Path("b"): 100, Path("c"): 10}
    assert find_undersized_dirs(counts) == {Path("c"): 10}


def test_find_undersized_dirs_returns_empty_when_balanced():
    counts = {Path("a"): 10, Path("b"): 11}
    assert find_undersized_dirs(counts) == {}


def test_find_undersized_dirs_handles_empty_input():
    assert find_undersized_dirs({}) == {}
