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
import time

from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from ..config import CrawlConfig
from ..sites import Site, face_query_param
from .base import (
    build_driver,
    build_query_url,
    get_scroll_position,
    highlight,
    pick_proxy,
    remove_duplicates,
    wait_and_click,
    wait_for_manual_captcha,
)

logger = logging.getLogger(__name__)

NAVER_SEARCH_URL = "https://search.naver.com/search.naver"

# NOTE: Verified against the live Naver image search DOM on 2026-08-29 and
# confirmed end-to-end (500 links collected, downloads succeeded). The old
# selector also required the thumbnail to sit inside a
# `div.tile_item._fe_image_tab_content_tile` wrapper, which under-matched
# real results on the current layout.
_THUMBNAIL_XPATH = '//img[@class="_fe_image_tab_content_thumbnail_image"]'


def collect_naver(keyword: str, config: CrawlConfig) -> list[str]:
    driver = build_driver(
        no_gui=config.no_gui, proxy=pick_proxy(config.proxy_list), user_data_dir=config.chrome_profile_dir
    )
    try:
        face_param = face_query_param(Site.NAVER) if config.face else ""
        url = build_query_url(NAVER_SEARCH_URL, "query", keyword, where="image", sm="tab_jum") + face_param
        driver.get(url)
        time.sleep(1)
        wait_for_manual_captcha(driver, config.no_gui)

        logger.info("Scrolling down")
        body = driver.find_element(By.TAG_NAME, "body")
        for _ in range(60):
            body.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.2)

        logger.info("Scraping links")
        imgs = driver.find_elements(By.XPATH, _THUMBNAIL_XPATH)

        links = []
        for img in imgs:
            src = img.get_attribute("src")
            if src and not src.startswith("d"):  # skip data: URI placeholders
                links.append(src)

        links = remove_duplicates(links)
        logger.info("Collect links done. Site: naver, Keyword: %s, Total: %d", keyword, len(links))
        return links
    finally:
        driver.quit()


def collect_naver_full(keyword: str, config: CrawlConfig) -> list[str]:
    """Full-resolution mode.

    NOTE: Verified against the live DOM on 2026-08-30. Bugs found and fixed:

    1. The viewer image now renders with two classes
       (`_fe_image_viewer_image_fallback_target _fe_image_viewer_main_image`),
       so the old exact-match `@class="..."` XPath no longer matched anything -
       switched to `contains()`.
    2. Re-reading the viewer image's src right after `Keys.RIGHT` mostly just
       re-read the *previous* image's still-current src, since Naver hadn't
       swapped it in yet - measured 0 new links across 60 iterations without a
       wait. Now waits (up to 5s) for the src to actually change from the last
       one collected before treating it as loaded - measured 80/80 with this
       in place. Also now stops once `config.limit` is reached, matching
       `collect_google_full` (previously ignored the limit entirely).
    3. End-of-results relied only on the scroll position going stale, same
       risk as `collect_google_full`'s NOTE (4). Naver adds a literal
       `disabled` class to the `_fe_image_viewer_next_button` link once the
       last image is shown, so that's now checked directly (same
       disabled-must-hold-for-15s / shortened post-disabled wait pattern as
       `collect_google_full`, for the same reasons); the scroll-based
       patience counter stays as a fallback. The disabled check is wrapped in
       its own `except StaleElementReferenceException` - see
       `collect_google_full`'s NOTE (6) for why an unguarded read here would
       be worse than not checking at all.
    """
    driver = build_driver(
        no_gui=config.no_gui, proxy=pick_proxy(config.proxy_list), user_data_dir=config.chrome_profile_dir
    )
    try:
        logger.info("[Full Resolution Mode]")
        face_param = face_query_param(Site.NAVER) if config.face else ""
        url = build_query_url(NAVER_SEARCH_URL, "query", keyword, where="image", sm="tab_jum") + face_param
        driver.get(url)
        time.sleep(1)
        wait_for_manual_captcha(driver, config.no_gui)

        body = driver.find_element(By.TAG_NAME, "body")
        logger.info("Scraping links")

        # Click the first image to open the full-resolution viewer.
        wait_and_click(driver, _THUMBNAIL_XPATH)
        time.sleep(1)

        links: list[str] = []
        limit = config.limit or 10000
        last_scroll = 0
        patience = 0
        last_src: str | None = None
        disabled_since: float | None = None
        disabled_confirm_seconds = 15
        xpath = '//img[contains(concat(" ", normalize-space(@class), " "), " _fe_image_viewer_image_fallback_target ")]'
        next_button_xpath = '//a[contains(concat(" ", normalize-space(@class), " "), " _fe_image_viewer_next_button ")]'

        while patience < 100 and len(links) < limit:
            try:
                # Wait for the viewer to actually advance to a new image (its src differs
                # from the last one we collected) instead of grabbing it right away - see
                # NOTE above for why that silently re-read the previous image most of the time.
                # Once disabled has been seen at least once, no new image is coming - shrink
                # the wait so confirming the end doesn't also drag in extra dead waiting.
                start = time.time()
                wait_timeout = 0.5 if disabled_since is not None else 5
                img = None
                src = None
                while time.time() - start < wait_timeout:
                    imgs = driver.find_elements(By.XPATH, xpath)
                    if imgs:
                        candidate = imgs[0].get_attribute("src")
                        if candidate and candidate != last_src:
                            src = candidate
                            img = imgs[0]
                            break
                    time.sleep(0.1)

                if src:
                    last_src = src
                    highlight(driver, img)
                    if src not in links:
                        links.append(src)
                        logger.info("%d: %s", len(links), src)
            except StaleElementReferenceException:
                pass

            try:
                next_buttons = driver.find_elements(By.XPATH, next_button_xpath)
                disabled = bool(next_buttons) and (
                    "disabled" in (next_buttons[0].get_attribute("class") or "").split()
                )
            except StaleElementReferenceException:
                disabled = False

            if disabled:
                if disabled_since is None:
                    disabled_since = time.time()
                elif time.time() - disabled_since >= disabled_confirm_seconds:
                    break
            else:
                disabled_since = None

            scroll = get_scroll_position(driver)
            if scroll == last_scroll:
                patience += 1
            else:
                patience = 0
                last_scroll = scroll

            body.send_keys(Keys.RIGHT)

        links = remove_duplicates(links)
        logger.info("Collect links done. Site: naver_full, Keyword: %s, Total: %d", keyword, len(links))
        return links
    finally:
        driver.quit()
