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
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from ..config import CrawlConfig
from ..sites import Site, face_query_param
from .base import (
    build_driver,
    build_query_url,
    get_scroll_position,
    highlight,
    is_captcha_page,
    pick_proxy,
    remove_duplicates,
    scroll_until_stable,
    wait_and_click,
    wait_for_manual_captcha,
)

logger = logging.getLogger(__name__)

GOOGLE_SEARCH_URL = "https://www.google.com/search"

# NOTE: Verified against the live Google Images DOM on 2026-08-29 (the previous
# `//div[@jsname="dTDiAc"]/div[@jsname="qQjpJ"]//img` selector no longer matched
# anything). Thumbnails are consistently served from this CDN host, which is
# far more stable than Google's obfuscated, frequently-rotated CSS classes.
#
# `encrypted-tbn` also serves small favicon/site-icon images (e.g. for "related searches"
# chips) unrelated to search results. Google sets a real, non-empty `alt` (the source
# page's title) on actual result thumbnails and always leaves `alt=""` on these decorative
# icons - confirmed against live results: every real thumbnail was >= 259px wide with a
# non-empty alt, every icon was <= 46px wide with alt="". Filtering on `alt` here is a
# structural distinction Google makes itself, rather than an inferred size cutoff.
_THUMBNAIL_XPATH = '//img[starts-with(@src, "https://encrypted-tbn") and string-length(@alt) > 0]'


def collect_google(keyword: str, config: CrawlConfig) -> list[str]:
    driver = build_driver(
        no_gui=config.no_gui, proxy=pick_proxy(config.proxy_list), user_data_dir=config.chrome_profile_dir
    )
    try:
        face_param = face_query_param(Site.GOOGLE) if config.face else ""
        url = build_query_url(GOOGLE_SEARCH_URL, "q", keyword, source="lnms", tbm="isch") + face_param
        driver.get(url)
        time.sleep(1)
        wait_for_manual_captcha(driver, config.no_gui)

        logger.info("Scrolling down")
        scroll_until_stable(driver)

        logger.info("Scraping links")
        imgs = driver.find_elements(By.XPATH, _THUMBNAIL_XPATH)
        links = remove_duplicates([img.get_attribute("src") for img in imgs])

        logger.info("Collect links done. Site: google, Keyword: %s, Total: %d", keyword, len(links))
        return links
    finally:
        driver.quit()


def collect_google_full(keyword: str, config: CrawlConfig) -> list[str]:
    """Full-resolution mode.

    NOTE: re-verified against the live DOM on 2026-08-30 (Google now redirects
    `tbm=isch` requests to a unified `?...&udm=2` layout, but the selectors
    below still match it - a JS-dispatched, untrusted `.click()` used while
    debugging this made it *look* completely broken, since Google's handlers
    only react to a real/trusted click). Two real bugs were found and fixed:

    1. Google appends a new `jsname="figiqf"` container for each image visited
       instead of replacing the old one, so `imgs[0]` (the first ever seen)
       kept returning the very first image forever after advancing past it -
       switched to `imgs[-1]`, the most recently added (current) one.
    2. `body.send_keys(Keys.RIGHT)` never actually advanced the viewer (the
       URL's image-id hash never changed, confirmed by direct testing) -
       sending keys to the `<body>` WebElement steals focus away from
       whatever the viewer needs focused to react to arrow keys. Switched to
       `ActionChains(driver).send_keys(Keys.RIGHT).perform()`, which sends the
       key to the page without focusing any particular element.
    3. Even with (2) fixed, advancing to the next image and immediately
       re-reading `imgs[-1]` mostly just re-read the *previous* image's still-
       current src, since Google hasn't appended the next `figiqf` container
       yet at that point - measured 72% of iterations wasted this way (86/120),
       silently skipping most images instead of collecting them. Now waits
       (up to 5s) for the src to actually change from the last one collected
       before treating it as loaded - measured 97.5% hit rate (117/120) with
       this in place.
    4. End-of-results was only detected by the scroll position going stale,
       but the background search-results page can keep scrolling (or just
       stay put) independently of the viewer, so this could run forever
       without ever hitting the real end. Google disables the "다음 이미지"
       (next image) button (`jsname="OCpkoe"`) once the last image is shown -
       confirmed live (`disabled` attribute appears exactly on the last
       image, collecting 101/101 links from a "dog" search before stopping) -
       so that's now checked directly and used as the primary stop condition;
       the scroll-based patience counter stays as a fallback. The disabled
       reading must hold for 15 continuous seconds before actually stopping,
       in case it's ever momentarily disabled mid-navigation rather than
       truly at the end. Once disabled has been seen at least once, the
       per-image wait in (3) drops from 5s to 0.5s (no new image is coming
       during this confirmation window), so those 15 seconds are the actual
       total dwell time at the end rather than being padded by extra 5s
       dead-waits stacked on top of it.
    5. A CAPTCHA can also appear mid-crawl, while already paging through
       images here - previously only checked once, right after the initial
       page load. Now also checked (cheaply) every iteration of this loop.
    6. The "다음 이미지" button read added for (4) sat outside the
       `except StaleElementReferenceException` block, so a staleness error
       there (observed live: `Message: stale element reference: stale
       element not found in the current frame`) escaped `collect_google_full`
       entirely and aborted the whole keyword/site task in `_run_task`
       instead of just this one iteration. Now caught locally.
    """
    driver = build_driver(
        no_gui=config.no_gui, proxy=pick_proxy(config.proxy_list), user_data_dir=config.chrome_profile_dir
    )
    try:
        logger.info("[Full Resolution Mode]")
        face_param = face_query_param(Site.GOOGLE) if config.face else ""
        url = build_query_url(GOOGLE_SEARCH_URL, "q", keyword, tbm="isch") + face_param
        driver.get(url)
        time.sleep(1)
        wait_for_manual_captcha(driver, config.no_gui)

        # Click the first image to get full resolution images.
        wait_and_click(driver, '//div[@jsname="dTDiAc"]')
        time.sleep(1)

        body = driver.find_element(By.TAG_NAME, "body")
        logger.info("Scraping links")

        links: list[str] = []
        limit = config.limit or 10000
        last_scroll = 0
        patience = 0
        max_patience = 100
        last_src: str | None = None
        disabled_since: float | None = None
        disabled_confirm_seconds = 15
        # Google renders a compressed thumbnail first, then overlaps it with the full image.
        xpath = '//div[@jsname="figiqf"]//img[not(contains(@src,"gstatic.com"))]'
        next_button_xpath = '//button[@jsname="OCpkoe"]'

        while len(links) < limit:
            try:
                # Wait for the viewer to actually advance to a new image (its src differs
                # from the last one we collected) instead of grabbing imgs[-1] right away -
                # see NOTE above for why that silently skipped most images. Once the next
                # button has already read as disabled at least once, there's no new image
                # left to wait for - shrink the wait so confirming the end (disabled_since
                # above) doesn't also drag an extra ~5s of dead waiting into every iteration.
                start = time.time()
                wait_timeout = 0.5 if disabled_since is not None else 5
                imgs: list = []
                src = None
                while time.time() - start < wait_timeout:
                    imgs = body.find_elements(By.XPATH, xpath)
                    if imgs:
                        candidate = imgs[-1].get_attribute("src")
                        if candidate and candidate != last_src:
                            src = candidate
                            break
                    time.sleep(0.1)

                if src:
                    last_src = src
                    highlight(driver, imgs[-1])
                    if src not in links:
                        links.append(src)
                        logger.info("%d: %s", len(links), src)
            except KeyboardInterrupt:
                break
            except StaleElementReferenceException:
                pass

            # A CAPTCHA can also appear mid-crawl while paging through images here, not just
            # on the initial page load - `is_captcha_page` is a single cheap DOM check (unlike
            # `wait_for_manual_captcha`'s own detection, which polls for up to 5s), so it's
            # safe to call every iteration and only pay the full wait when one is actually
            # showing.
            if is_captcha_page(driver):
                wait_for_manual_captcha(driver, config.no_gui)

            try:
                next_buttons = driver.find_elements(By.XPATH, next_button_xpath)
                disabled = bool(next_buttons) and (
                    next_buttons[0].get_attribute("disabled") is not None
                    or next_buttons[0].get_attribute("aria-disabled") == "true"
                )
            except StaleElementReferenceException:
                # Treat an unreadable button the same as "not disabled" - just means this
                # iteration's reading doesn't count towards disabled_confirm_seconds, not
                # that the whole task should fail (see NOTE (6) above).
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
            if patience >= max_patience:
                break

            ActionChains(driver).send_keys(Keys.RIGHT).perform()

        links = remove_duplicates(links)
        logger.info("Collect links done. Site: google_full, Keyword: %s, Total: %d", keyword, len(links))
        return links
    finally:
        driver.quit()
