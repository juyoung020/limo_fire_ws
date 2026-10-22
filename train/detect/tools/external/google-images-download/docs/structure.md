# Workflow

[Documentation Homepage](index.md) | [GitHub Repository](https://github.com/hardikvasa/google-images-download)

Below diagram represents the algorithm logic used to download images.

![Workflow Diagram](http://www.zseries.in/flow-chart.png)

## How It Works

1. **Parse Arguments**: The tool reads command-line arguments or programmatic parameters
2. **Build Search URL**: Constructs a Google Images search URL with all specified filters (size, color, format, etc.)
3. **Launch Browser**: Uses Selenium with Chrome to load the Google Images page
4. **Execute JavaScript**: Scrolls the page and loads additional images by executing JavaScript
5. **Extract Image URLs**: Parses the page source to extract all image URLs
6. **Download Images**: Downloads each image to the specified directory
7. **Handle Errors**: Tracks and reports any download errors
8. **Save Metadata** (optional): Extracts and saves image metadata to JSON files

## Key Features

- **Automatic Scrolling**: The tool automatically scrolls the Google Images page to load more results
- **JavaScript Execution**: Uses Selenium to handle Google's dynamic content loading
- **Anti-Detection**: Implements measures to avoid being detected as a bot
- **Error Handling**: Gracefully handles network errors and invalid images
- **Parallel Processing**: Can handle multiple keywords in a single execution
- **Filename Sanitization**: Automatically handles special characters in filenames
