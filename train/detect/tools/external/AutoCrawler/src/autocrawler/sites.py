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

from enum import StrEnum


class Site(StrEnum):
    """Supported image search sites."""

    GOOGLE = "google"
    NAVER = "naver"


def face_query_param(site: Site) -> str:
    """Query string suffix that restricts results to faces, per site."""
    if site is Site.GOOGLE:
        return "&tbs=itp:face"
    return "&face=1"
