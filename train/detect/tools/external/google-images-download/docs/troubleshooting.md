# Troubleshooting Errors/Issues

[Documentation Homepage](index.md) | [GitHub Repository](https://github.com/hardikvasa/google-images-download)

## Chrome/Chromium and Selenium Setup

### Chrome is Required

Chrome/Chromium browser is **mandatory** for this tool. Google Images requires JavaScript execution, so all downloads use Selenium with Chrome.

**Installation by OS:**

- **Linux**: `apt-get install chromium-browser` or `apt-get install google-chrome-stable`
- **macOS**: `brew install --cask google-chrome`
- **Windows**: Download from [https://www.google.com/chrome/](https://www.google.com/chrome/)

### Chromedriver Management

The tool automatically downloads and manages chromedriver using `webdriver-manager`. **You do not need to manually download or configure chromedriver.**

If you encounter chromedriver issues, the tool will attempt to download the correct version automatically.

### Using Custom Chrome Binary Location

If you have Chrome/Chromium installed in a non-standard location (e.g., Playwright's Chromium), the `--chromedriver` argument alone is not sufficient. The tool does not currently support specifying a custom Chrome binary location through command-line arguments.

**Workaround for custom Chrome binary:**

If using the library programmatically, you can patch the Chrome options:

```python
from google_images_download import google_images_download as gid
from selenium.webdriver.chrome.options import Options

# Patch the download_extended_page method
original_method = gid.googleimagesdownload.download_extended_page

def patched_method(self, url, chromedriver):
    # Import inside function to avoid circular imports
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options

    options = Options()
    options.binary_location = '/path/to/your/chrome/binary'  # Set custom location
    options.add_argument('--no-sandbox')
    options.add_argument('--headless=new')
    options.add_argument('--disable-dev-shm-usage')

    service = Service(chromedriver) if chromedriver else Service()
    browser = webdriver.Chrome(service=service, options=options)

    browser.get(url)
    # ... rest of the scrolling logic ...
    html = browser.page_source
    browser.quit()
    return html

gid.googleimagesdownload.download_extended_page = patched_method

# Now use normally
response = gid.googleimagesdownload()
response.download({"keywords": "test", "limit": 5})
```

### Linux-specific Chrome Installation

- **CentOS or Amazon Linux**: [Installation Guide](https://intoli.com/blog/installing-google-chrome-on-centos/)
- **Ubuntu**: [Installation Guide](https://askubuntu.com/questions/510056/how-to-install-google-chrome)

## SSL Errors

If you see SSL errors on Mac for Python 3, go to:

**Finder → Applications → Python 3 → Click on 'Install Certificates.command'** and run the file.

Alternatively, use this command:

```bash
cd /Applications/Python\ 3.*/
./Install\ Certificates.command
```

## googleimagesdownload: command not found

While using the above commands, if you get `Error: -bash: googleimagesdownload: command not found`, you need to set the correct path variable.

To get the details of the repo, run:

```bash
pip show -f google_images_download
```

You will get a result like:

```
Location: /Library/Frameworks/Python.framework/Versions/3.x/lib/python3.x/site-packages
Files:
  ../../../bin/googleimagesdownload
```

Together they make: `/Library/Frameworks/Python.framework/Versions/3.x/bin` which you need to add to your PATH:

```bash
export PATH="/Library/Frameworks/Python.framework/Versions/3.x/bin:$PATH"
```

Add this line to your `~/.bashrc` or `~/.zshrc` to make it permanent.

## Permission denied creating directory 'downloads'

When you run the command, it downloads the images in the current directory. If you get a permission denied error for creating the `downloads` directory, move to a directory where you have write permission and then run the command again.

## Permission denied while installing the library

On macOS and Linux, when you get permission denied when installing the library using pip, try doing a user install:

```bash
pip install google_images_download --user
```

You can also run pip install as a superuser with `sudo pip install google_images_download`, but it is not generally a good idea because it can cause issues with your system-level packages.

## Session not created: cannot find Chrome binary

If you get this error:

```
Message: session not created: cannot find Chrome binary
```

This means Chrome/Chromium is not installed or not in your system PATH. Install Chrome using the instructions at the top of this page.

## urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]

[Related Issue #140](https://github.com/hardikvasa/google-images-download/issues/140)

Use the below command to install the SSL certificate on your machine:

```bash
cd /Applications/Python\ 3.*/
./Install\ Certificates.command
```

## ModuleNotFoundError: No module named 'selenium'

If you get this error, install the selenium dependency:

```bash
pip install selenium webdriver-manager
```

Or reinstall google-images-download:

```bash
pip install --upgrade --force-reinstall google_images_download
```

## Images fail to download / 0 images downloaded

This usually happens when:

1. **Network connectivity issues**: Check your internet connection
2. **Google blocking automated requests**: Try adding a delay with `--delay 1`
3. **Invalid search parameters**: Verify your keywords and filters are correct
4. **Rate limiting**: Reduce `--limit` or add delays between downloads

If the issue persists, please [open an issue](https://github.com/hardikvasa/google-images-download/issues) with your command and error output.
