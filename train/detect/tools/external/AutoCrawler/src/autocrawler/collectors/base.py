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
import random
import shutil
import time
from pathlib import Path
from urllib.parse import quote_plus

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

logger = logging.getLogger(__name__)


def pick_proxy(proxy_list: list[str] | None) -> str | None:
    return random.choice(proxy_list) if proxy_list else None


def build_driver(no_gui: bool = False, proxy: str | None = None, user_data_dir: Path | None = None) -> webdriver.Chrome:
    options = Options()
    options.add_argument("--no-sandbox")  # To maintain user cookies
    options.add_argument("--disable-dev-shm-usage")
    if no_gui:
        options.add_argument("--headless")
    if proxy:
        options.add_argument(f"--proxy-server={proxy}")
    if user_data_dir:
        # Persist cookies (e.g. a manually-solved CAPTCHA) across runs instead of
        # starting from a throwaway profile every time.
        user_data_dir.mkdir(parents=True, exist_ok=True)
        options.add_argument(f"--user-data-dir={user_data_dir}")

    # Prefer a chromedriver already on PATH (e.g. installed via `brew install --cask chromedriver`)
    # over downloading a separate copy with webdriver-manager: on macOS, Gatekeeper trust is
    # per-binary, so a driver the user has already approved keeps working, while a fresh
    # webdriver-manager download under ~/.wdm would need to be approved all over again.
    driver_path = shutil.which("chromedriver") or ChromeDriverManager().install()
    driver = webdriver.Chrome(service=Service(driver_path), options=options)

    browser_version = str(driver.capabilities.get("browserVersion", "unknown"))
    chrome_caps = driver.capabilities.get("chrome", {})
    chromedriver_version = str(chrome_caps.get("chromedriverVersion", "unknown")).split(" ")[0]

    if browser_version.split(".")[0] != chromedriver_version.split(".")[0]:
        logger.warning(
            "Chrome (%s) / chromedriver (%s) major version mismatch", browser_version, chromedriver_version
        )

    return driver


def build_query_url(base_url: str, query_param: str, query: str, **params: str) -> str:
    """Build a search URL, percent-encoding the free-text query.

    The original crawler inserted keywords into URLs unescaped, which broke
    searches for keywords containing spaces or characters like `&`/`#`.
    """
    parts = [f"{query_param}={quote_plus(query)}"] + [f"{key}={value}" for key, value in params.items()]
    return f"{base_url}?{'&'.join(parts)}"


def get_scroll_position(driver: webdriver.Chrome) -> int:
    return driver.execute_script("return window.pageYOffset;")


def scroll_until_stable(driver: webdriver.Chrome, max_patience: int = 50, delay: float = 0.2) -> None:
    """Scroll the page down until the scroll position stops advancing."""
    body = driver.find_element(By.TAG_NAME, "body")

    last_scroll = 0
    patience = 0

    while patience < max_patience:
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(delay)

        scroll = get_scroll_position(driver)
        if scroll == last_scroll:
            patience += 1
        else:
            patience = 0
            last_scroll = scroll


def highlight(driver: webdriver.Chrome, element: WebElement) -> None:
    driver.execute_script(
        "arguments[0].setAttribute('style', arguments[1]);",
        element,
        "background: yellow; border: 2px solid red;",
    )


def wait_and_click(driver: webdriver.Chrome, xpath: str, timeout: float = 15) -> WebElement:
    # Sometimes click fails unreasonably, so this tries at all cost.
    try:
        element = WebDriverWait(driver, timeout).until(EC.element_to_be_clickable((By.XPATH, xpath)))
        element.click()
        highlight(driver, element)
        return element
    except Exception:
        logger.warning("Click timed out - %s; refreshing browser...", xpath)
        driver.refresh()
        time.sleep(2)
        return wait_and_click(driver, xpath, timeout)


def remove_duplicates(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


_CAPTCHA_WIDGET_SELECTOR = (
    "iframe[src*='recaptcha' i], iframe[title*='recaptcha' i], div.g-recaptcha, #captcha-form"
)


def is_captcha_page(driver: webdriver.Chrome) -> bool:
    """Check for an actually-rendered CAPTCHA challenge, not just incidental page text.

    A plain substring search over `page_source` (e.g. for "captcha") false-positives on
    pages that merely load a reCAPTCHA script/badge for unrelated widgets (login forms,
    ads, etc.) without showing a challenge. Require the real Google block URL or a
    visible CAPTCHA widget element instead.
    """
    if "/sorry/" in driver.current_url:
        return True
    return bool(driver.find_elements(By.CSS_SELECTOR, _CAPTCHA_WIDGET_SELECTOR))


def wait_for_manual_captcha(
    driver: webdriver.Chrome, no_gui: bool, poll_seconds: float = 5.0, poll_interval: float = 0.5
) -> None:
    """If the current page looks like a bot-check/CAPTCHA page, pause and ask a human to clear it.

    The CAPTCHA widget (a reCAPTCHA iframe) loads asynchronously and may not be in the DOM
    yet right after navigation, so this polls for a few seconds rather than checking once -
    otherwise a real CAPTCHA can be missed and the crawler moves on before it even renders.
    """
    deadline = time.time() + poll_seconds
    detected = is_captcha_page(driver)
    while not detected and time.time() < deadline:
        time.sleep(poll_interval)
        detected = is_captcha_page(driver)

    if not detected:
        return

    if no_gui:
        logger.error("CAPTCHA detected, but running headless (--no_gui) - cannot prompt for a manual solve.")
        return

    logger.warning("CAPTCHA detected at %s", driver.current_url)
    try:
        input("Solve the CAPTCHA in the open Chrome window, then press Enter here to continue...\n")
    except EOFError:
        # No interactive stdin here - e.g. this collector is running inside a worker
        # process spawned for --threads > 1. Fail this task cleanly instead of letting
        # a raw EOFError bubble up; rerun with --threads 1 to solve CAPTCHAs by hand.
        logger.error("CAPTCHA needs a human, but this task has no interactive terminal (are --threads > 1?).")
        raise RuntimeError("CAPTCHA blocked this task and no interactive terminal was available to solve it.")
