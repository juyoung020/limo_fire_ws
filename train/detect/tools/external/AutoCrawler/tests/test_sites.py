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

from autocrawler.collectors.base import build_query_url
from autocrawler.sites import Site, face_query_param


def test_face_query_param_google():
    assert face_query_param(Site.GOOGLE) == "&tbs=itp:face"


def test_face_query_param_naver():
    assert face_query_param(Site.NAVER) == "&face=1"


def test_build_query_url_encodes_spaces_and_special_characters():
    url = build_query_url("https://example.com/search", "q", "black cat & dog", tbm="isch")
    assert url == "https://example.com/search?q=black+cat+%26+dog&tbm=isch"


def test_build_query_url_without_extra_params():
    url = build_query_url("https://example.com/search", "query", "cat")
    assert url == "https://example.com/search?query=cat"
