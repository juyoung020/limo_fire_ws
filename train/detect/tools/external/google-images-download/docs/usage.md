# Usage

[Documentation Homepage](index.md) | [GitHub Repository](https://github.com/hardikvasa/google-images-download)

## Using the library from Command Line Interface

If installed via pip, use the following command:

```bash
googleimagesdownload [Arguments...]
```

If downloaded via the UI, unzip the file downloaded, go to the 'google_images_download' directory and use one of the below commands:

```bash
python3 google_images_download.py [Arguments...]
# OR
python google_images_download.py [Arguments...]
```

## Using the library from another python file

If you would want to use this library from another python file, you could use it as shown below:

```python
from google_images_download import google_images_download

response = google_images_download.googleimagesdownload()
absolute_image_paths = response.download({<Arguments...>})
```

The `download()` method returns a tuple: `(paths_dict, error_count)` where:
- `paths_dict`: Dictionary mapping keywords to lists of downloaded image paths
- `error_count`: Number of errors encountered during download

## Basic Example

```python
from google_images_download import google_images_download

response = google_images_download.googleimagesdownload()

arguments = {
    "keywords": "puppies,kittens",
    "limit": 10,
    "print_urls": True
}

paths, errors = response.download(arguments)
print(f"Downloaded images: {paths}")
print(f"Errors: {errors}")
```

For more examples, see the [Examples](examples.md) page.
