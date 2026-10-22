# Input Arguments

[Documentation Homepage](index.md) | [GitHub Repository](https://github.com/hardikvasa/google-images-download)

This page provides a complete reference of all arguments/parameters available in google-images-download.

## Arguments Reference

| Argument | Short | Description |
|----------|-------|-------------|
| `config_file` | `cf` | Pass arguments via a config file (JSON format). If this argument is present, command line arguments will be discarded. See [config file format](examples.md#config-file-format) for details. |
| `keywords` | `k` | Keywords/key phrases to search for. For multiple keywords, wrap in quotes. **Tips:** Type keyword normally for best match, wrap in double quotes ("") for exact phrase, use OR between words for either/or search, use minus (-) to exclude words. |
| `keywords_from_file` | `kf` | File name to import keywords from. Add one keyword per line. Only `.txt` or `.csv` files allowed. |
| `prefix_keywords` | `pk` | Words added before main keyword. Example: if keyword is 'car' and prefix is 'red,blue', searches 'red car' and 'blue car'. |
| `suffix_keywords` | `sk` | Words added after main keyword. Example: if keyword is 'car' and suffix is 'red,blue', searches 'car red' and 'car blue'. |
| `limit` | `l` | Number of images to download. Defaults to 100 if not specified. |
| `related_images` | `ri` | Downloads images from related keywords shown on Google Images. No value needed, just add the flag. **Note:** Can download hundreds/thousands of additional images. |
| `format` | `f` | Image format/extension. **Values:** `jpg`, `gif`, `png`, `bmp`, `svg`, `webp`, `ico`, `raw` |
| `color` | `co` | Color filter. **Values:** `red`, `orange`, `yellow`, `green`, `teal`, `blue`, `purple`, `pink`, `white`, `gray`, `black`, `brown` |
| `color_type` | `ct` | Color type filter. **Values:** `full-color`, `black-and-white`, `transparent` |
| `usage_rights` | `r` | Usage rights/license filter. **Values:** `labeled-for-reuse-with-modifications`, `labeled-for-reuse`, `labeled-for-noncommercial-reuse-with-modification`, `labeled-for-nocommercial-reuse` |
| `size` | `s` | Relative size of images. **Values:** `large`, `medium`, `icon`, `>400*300`, `>640*480`, `>800*600`, `>1024*768`, `>2MP`, `>4MP`, `>6MP`, `>8MP`, `>10MP`, `>12MP`, `>15MP`, `>20MP`, `>40MP`, `>70MP` |
| `exact_size` | `es` | Exact image resolution. Format: `<width>,<height>` (e.g., `-es 1024,768`). **Note:** Cannot use both `size` and `exact_size` in same query. |
| `aspect_ratio` | `a` | Aspect ratio filter. **Values:** `tall`, `square`, `wide`, `panoramic` |
| `type` | `t` | Image type filter. **Values:** `face`, `photo`, `clip-art`, `line-drawing`, `animated` |
| `time` | `w` | Time uploaded/indexed. **Values:** `past-24-hours`, `past-7-days`, `past-month`, `past-year` |
| `time_range` | `wr` | Custom time range. Format: `'{"time_min":"MM/DD/YYYY","time_max":"MM/DD/YYYY"}'` |
| `delay` | `d` | Time to wait between downloading two images (in seconds, decimal values allowed). |
| `url` | `u` | Download images from a Google Images search URL. Paste the browser URL from Google Images page. |
| `single_image` | `x` | Download one image from its complete (absolute) URL. |
| `output_directory` | `o` | Main directory name for downloads. Defaults to 'downloads' in current path. |
| `image_directory` | `i` | Sub-directory inside output_directory. Defaults to keyword name. |
| `no_directory` | `n` | Download images directly in output_directory without sub-directories. No value needed. |
| `proxy` | `px` | Proxy server setting for all requests. Format: `IP:Port` |
| `similar_images` | `si` | Reverse image search. Downloads images similar to the provided image URL. |
| `specific_site` | `ss` | Download images only from a specific website/domain. |
| `print_urls` | `p` | Print image URLs to console (useful for debugging). No value needed. |
| `print_size` | `ps` | Print actual image size to console. No value needed. |
| `print_paths` | `pp` | Print absolute paths of downloaded images. Returns list when called from Python. No value needed. |
| `metadata` | `m` | Print image metadata (size, origin, attributes, description, URL). No value needed. |
| `extract_metadata` | `e` | Save metadata of all downloaded images to JSON file in `logs/` directory. No value needed. |
| `socket_timeout` | `st` | Socket connection timeout in seconds. Default is 10 seconds. Increase for slow connections. |
| `thumbnail` | `th` | Download image thumbnails in addition to full images. Saved in sub-directories. No value needed. |
| `thumbnail_only` | `tho` | Download only thumbnails without full-size images. Saved in sub-directories. No value needed. |
| `language` | `la` | Language filter for search results. **Values:** `Arabic`, `Chinese (Simplified)`, `Chinese (Traditional)`, `Czech`, `Danish`, `Dutch`, `English`, `Estonian`, `Finnish`, `French`, `German`, `Greek`, `Hebrew`, `Hungarian`, `Icelandic`, `Italian`, `Japanese`, `Korean`, `Latvian`, `Lithuanian`, `Norwegian`, `Portuguese`, `Polish`, `Romanian`, `Russian`, `Spanish`, `Swedish`, `Turkish` |
| `prefix` | `pr` | Prefix to add before image filenames (useful for image identification). |
| `chromedriver` | `cd` | Path to chromedriver executable. Usually not needed as `webdriver-manager` handles this automatically. Format: `path/to/chromedriver` (Windows: `C:\\path\\to\\chromedriver.exe`) |
| `safe_search` | `sa` | Enable Safe Search filter. Filter is OFF by default. No value needed. |
| `no_numbering` | `nn` | Disable ordered numbering prefix on downloaded images. No value needed. |
| `offset` | `of` | Skip first N images before downloading. Must be less than limit value. |
| `save_source` | `is` | Create text file with list of downloaded images and their source page URLs. Takes filename as value. |
| `no_download` | `nd` | Print URLs without downloading images. Useful for getting image URLs only. No value needed. |
| `silent_mode` | `sil` | Suppress all notification messages on terminal. Overrides all print arguments. No value needed. |
| `ignore_urls` | `iu` | Skip images whose URLs contain specified strings. Format: comma-separated values (e.g., `wikipedia.org,wikimedia.org`) |
| `help` | `h` | Show help message with usage information. |

## Important Notes

- **Mandatory Parameter:** If `single_image` or `url` parameter is not present, then `keywords` is mandatory. No other parameters are mandatory.
- **Chrome Requirement:** Chrome/Chromium browser is required for all downloads. Selenium is automatically installed as a dependency.
- **Chromedriver:** Automatically managed by `webdriver-manager`. Manual setup is rarely needed.

## Usage Examples

See the [Examples](examples.md) page for detailed usage examples of these arguments.
