# Examples

[Documentation Homepage](index.md) | [GitHub Repository](https://github.com/hardikvasa/google-images-download) | [Input Arguments](arguments.md)

## Config File Format

You can either pass the arguments directly from the command line or through a config file. Below is a sample of how a config file looks.

You can pass more than one record through a config file. The code will iterate through each record and download images based on the arguments passed.

```json
{
    "Records": [
        {
            "keywords": "apple",
            "limit": 5,
            "color": "green",
            "print_urls": true
        },
        {
            "keywords": "universe",
            "limit": 15,
            "size": "large",
            "print_urls": true
        }
    ]
}
```

## Code Example - Importing the Library

If you are calling this library from another Python file:

```python
from google_images_download import google_images_download   # importing the library

response = google_images_download.googleimagesdownload()   # class instantiation

arguments = {
    "keywords": "Polar bears,baloons,Beaches",
    "limit": 20,
    "print_urls": True
}

paths = response.download(arguments)   # passing the arguments to the function
print(paths)   # printing absolute paths of the downloaded images
```

## Command Line Examples

### Using a config file

If you are passing arguments from a config file, simply pass the config_file argument with the name of your JSON file:

```bash
googleimagesdownload -cf example.json
```

### Simple keyword search with limit

```bash
googleimagesdownload --keywords "Polar bears, baloons, Beaches" --limit 20
```

### Using shorthand arguments

```bash
googleimagesdownload -k "Polar bears, baloons, Beaches" -l 20
```

### Using Suffix Keywords

Suffix Keywords allow you to specify words after the main keywords. For example, if `keyword = car` and `suffix keyword = 'red,blue'`, it will first search for `car red` and then `car blue`:

```bash
googleimagesdownload --k "car" -sk 'red,blue,white' -l 10
```

### Download images with specific format

```bash
googleimagesdownload --keywords "logo" --format svg
```

### Using color filters

```bash
googleimagesdownload -k "playground" -l 20 -co red
```

### Non-English keywords

```bash
googleimagesdownload -k "北极熊" -l 5
```

### Download from a Google Images URL

```bash
googleimagesdownload -k "sample" -u <google images page URL>
```

### Save to specific directory

Instead of saving to 'downloads', save to a custom directory:

```bash
googleimagesdownload -k "boat" -o "boat_images"
```

### Download a single image by URL

```bash
googleimagesdownload --keywords "baloons" --single_image <URL of the image>
```

### Size and type constraints

```bash
googleimagesdownload --keywords "baloons" --size medium --type animated
```

### Specific usage rights

```bash
googleimagesdownload --keywords "universe" --usage_rights labeled-for-reuse
```

### Specific color type

```bash
googleimagesdownload --keywords "flowers" --color_type black-and-white
```

### Specific aspect ratio

```bash
googleimagesdownload --keywords "universe" --aspect_ratio panoramic
```

### Reverse Image Search

Download images similar to the image URL provided:

```bash
googleimagesdownload -si <image url> -l 10
```

### Download from specific website

```bash
googleimagesdownload --keywords "universe" --specific_site example.com
```

### Using chromedriver path (if needed)

If the automatic chromedriver management fails, you can specify the path manually:

```bash
googleimagesdownload -k "puppies" -l 10 --chromedriver /path/to/chromedriver
```

On Windows:

```bash
googleimagesdownload -k "puppies" -l 10 --chromedriver C:\\path\\to\\chromedriver.exe
```

### Print URLs without downloading

```bash
googleimagesdownload -k "universe" -l 10 --no_download --print_urls
```

### Extract metadata to JSON

```bash
googleimagesdownload -k "nature" -l 20 --extract_metadata
```

This creates a JSON file in the `logs/` directory with metadata for all downloaded images.

### Download with delay

Add a delay between downloads to avoid rate limiting:

```bash
googleimagesdownload -k "sunset" -l 50 --delay 1
```

### Related images

Download images from related keywords (can download hundreds of additional images):

```bash
googleimagesdownload -k "car" -l 10 --related_images
```

**Note:** The images are downloaded in their own sub-directories inside the main directory (either the one you provided with `-o` or in 'downloads') in the same folder you are in.

## Library Extensions

### Cleaning Corrupt Images

The downloading algorithm does a good job of keeping out corrupt images, but it's not perfect. Below script will help clean corrupt image files. This script was ideated by @devajith in [Issue 81](https://github.com/hardikvasa/google-images-download/issues/81):

```python
import os
from PIL import Image

img_dir = r"path/to/downloads/directory"
for filename in os.listdir(img_dir):
    try:
        with Image.open(img_dir + "/" + filename) as im:
            print('ok')
    except:
        print(img_dir + "/" + filename)
        os.remove(img_dir + "/" + filename)
```
