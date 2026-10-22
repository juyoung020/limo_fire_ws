# Installation

[Documentation Homepage](index.md) | [GitHub Repository](https://github.com/hardikvasa/google-images-download)

You can use **one of the below methods** to download and use this repository.

## Requirements

**Chrome/Chromium browser is required** for this tool to work. Google Images now requires JavaScript execution, so Selenium with Chrome is mandatory for all downloads.

### Installing Chrome/Chromium

- **Linux**: `apt-get install chromium-browser` or `apt-get install google-chrome-stable`
- **macOS**: `brew install --cask google-chrome`
- **Windows**: Download from [https://www.google.com/chrome/](https://www.google.com/chrome/)

**Note:** The tool will automatically download and manage the chromedriver using `webdriver-manager`. No manual chromedriver setup is required.

## Install using pip

```bash
pip install google_images_download
```

This will automatically install the required dependencies including `selenium` and `webdriver-manager`.

## Manually install using CLI

```bash
git clone https://github.com/hardikvasa/google-images-download.git
cd google-images-download
pip install -e .
```

## Manually install using UI

1. Go to the [repo on GitHub](https://github.com/hardikvasa/google-images-download)
2. Click on 'Clone or Download'
3. Click on 'Download ZIP' and save it on your local disk
4. Extract the ZIP file
5. Navigate to the extracted directory and run:
   ```bash
   pip install -e .
   ```

## Verify Installation

After installation, verify it works:

```bash
googleimagesdownload --keywords "test" --limit 1
```

You should see the tool download one image to the `downloads/test/` directory.
