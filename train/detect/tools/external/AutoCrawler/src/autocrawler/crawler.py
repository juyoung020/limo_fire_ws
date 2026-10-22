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
import shutil
import signal
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path

from .collectors.base import build_driver, wait_for_manual_captcha
from .collectors.google import collect_google, collect_google_full
from .collectors.naver import collect_naver, collect_naver_full
from .config import CrawlConfig, CrawlTask
from .downloader import download_images
from .files import all_dirs, all_files, get_keywords
from .sites import Site

logger = logging.getLogger(__name__)


def _init_worker() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Worker processes are spawned fresh (the default start method on macOS/Windows),
    # so they don't inherit the main process's logging.basicConfig() call - without
    # this, every logger.info() from a worker (link collection, download progress,
    # "Done") is silently dropped instead of printed.
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def warm_up_chrome_profile(config: CrawlConfig) -> None:
    """Open one visible Chrome window against Google so a human can clear any CAPTCHA
    once, before the real (possibly multi-process) crawl starts.

    Only Google needs this - Naver hasn't been observed to CAPTCHA-block this crawler.
    The resulting cookies land in `config.chrome_profile_dir`; each worker task then
    crawls with its own copy of that profile (see `_task_profile_dir`), so multiple
    Chrome processes never fight over the same profile directory lock.
    """
    if not config.do_google:
        return

    driver = build_driver(no_gui=False, user_data_dir=config.chrome_profile_dir)
    try:
        driver.get("https://www.google.com/search?q=warmup&tbm=isch")
        time.sleep(1)
        wait_for_manual_captcha(driver, no_gui=False)
    finally:
        driver.quit()


def _task_profile_dir(config: CrawlConfig) -> Path:
    """A private copy of `config.chrome_profile_dir` for one task to use.

    Chrome refuses to run two instances against the same profile directory at once,
    so concurrent tasks each get their own copy (carrying over the warmed-up cookies)
    instead of sharing - and racing on - a single directory.
    """
    task_dir = Path(tempfile.mkdtemp(prefix="autocrawler-chrome-"))
    if config.chrome_profile_dir.exists():
        # Chrome's own lock files (sockets/pipes, not plain files) can't be copied and
        # don't need to be - a new instance recreates them itself.
        ignore = shutil.ignore_patterns("Singleton*", "RunningChromeVersion")
        shutil.copytree(config.chrome_profile_dir, task_dir, dirs_exist_ok=True, ignore=ignore)
    return task_dir


def _collect_links(task: CrawlTask, config: CrawlConfig) -> list[str]:
    task_profile_dir = _task_profile_dir(config)
    task_config = replace(config, chrome_profile_dir=task_profile_dir)
    try:
        if task.site is Site.GOOGLE:
            if task.full_resolution:
                return collect_google_full(task.keyword, task_config)
            return collect_google(task.keyword, task_config)
        if task.full_resolution:
            return collect_naver_full(task.keyword, task_config)
        return collect_naver(task.keyword, task_config)
    finally:
        shutil.rmtree(task_profile_dir, ignore_errors=True)


def _run_task(task: CrawlTask, config: CrawlConfig) -> None:
    try:
        logger.info("Collecting links... %s from %s", task.keyword, task.site.value)
        links = _collect_links(task, config)
    except Exception as exc:  # noqa: BLE001 - one bad task shouldn't kill the pool
        logger.error("Error occurred while collecting %s:%s - %s", task.site.value, task.keyword, exc)
        return

    logger.info("Downloading images from collected links... %s from %s", task.keyword, task.site.value)
    download_images(task.keyword, links, task.site, config.download_path, max_count=config.limit)

    keyword_dir = config.download_path / task.keyword.replace('"', "")
    (keyword_dir / f"{task.site.value}_done").touch()
    logger.info("Done %s : %s", task.site.value, task.keyword)


def _build_tasks(config: CrawlConfig) -> list[CrawlTask]:
    keywords = get_keywords(config.keywords_file)
    tasks: list[CrawlTask] = []

    for keyword in keywords:
        keyword_dir = config.download_path / keyword.replace('"', "")
        google_done = (keyword_dir / "google_done").exists()
        naver_done = (keyword_dir / "naver_done").exists()

        if google_done and naver_done and config.skip_already_exist:
            logger.info("Skipping done task %s", keyword_dir)
            continue

        if config.do_google and not google_done:
            tasks.append(CrawlTask(keyword, Site.GOOGLE, config.full_resolution))
        if config.do_naver and not naver_done:
            tasks.append(CrawlTask(keyword, Site.NAVER, config.full_resolution))

    return tasks


def find_undersized_dirs(counts: dict[Path, int], threshold: float = 0.5) -> dict[Path, int]:
    """Directories with fewer than `threshold` of the average file count."""
    if not counts:
        return {}
    average = sum(counts.values()) / len(counts)
    return {directory: count for directory, count in counts.items() if count < average * threshold}


def imbalance_check(download_path: Path) -> None:
    logger.info("Data imbalance checking...")

    counts = {directory: len(all_files(directory)) for directory in all_dirs(download_path)}
    for directory, count in counts.items():
        logger.info("dir: %s, file_count: %d", directory, count)

    undersized = find_undersized_dirs(counts)
    if not undersized:
        logger.info("Data imbalance not detected.")
        return

    logger.info("Data imbalance detected.")
    logger.info("Below keywords have smaller than 50%% of average file count.")
    logger.info("I recommend you to remove these directories and re-download for that keyword.")
    logger.info("_________________________________")
    logger.info("Too small file count directories:")
    for directory, count in undersized.items():
        logger.info("dir: %s, file_count: %d", directory, count)

    answer = input("Remove directories above? (y/n)\n")
    if answer == "y":
        logger.info("Removing too small file count directories...")
        for directory in undersized:
            shutil.rmtree(directory)
            logger.info("Removed %s", directory)
        logger.info("Now re-run this program to re-download removed files. (with skip_already_exist=True)")


def run(config: CrawlConfig) -> None:
    config.download_path.mkdir(parents=True, exist_ok=True)

    if not config.no_gui:
        warm_up_chrome_profile(config)

    tasks = _build_tasks(config)
    worker = partial(_run_task, config=config)

    if config.n_processes <= 1:
        # Run in-process rather than in a worker subprocess: a subprocess's stdin isn't
        # connected to the terminal, so `input()` (used to prompt for a manual CAPTCHA
        # solve) would fail immediately with EOFError instead of waiting for the user.
        try:
            for task in tasks:
                worker(task)
        except KeyboardInterrupt:
            pass
    else:
        executor = ProcessPoolExecutor(max_workers=config.n_processes, initializer=_init_worker)
        try:
            try:
                list(executor.map(worker, tasks))
            except KeyboardInterrupt:
                # Workers ignore SIGINT (see _init_worker), so Ctrl+C only raises here in the
                # main process - cancelling futures would just skip ones that haven't started
                # yet, then block waiting for whatever's already running to finish on its own
                # (which can take a very long time). Terminate the worker processes directly:
                # they don't ignore SIGTERM.
                logger.warning("Interrupted - terminating worker processes...")
                for process in executor._processes.values():
                    process.terminate()
        finally:
            executor.shutdown(wait=True)

    logger.info("Task ended. Pool join.")
    imbalance_check(config.download_path)
    logger.info("End Program")
