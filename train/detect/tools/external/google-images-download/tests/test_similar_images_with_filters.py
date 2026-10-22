"""
Test for issue #119: Other options are ignored when -si (similar_images) is used.
This test verifies that filter parameters (size, format, color, etc.) are properly
included in the URL when using similar_images option.
"""

from google_images_download import google_images_download
import unittest


class TestSimilarImagesWithFilters(unittest.TestCase):

    def setUp(self):
        self.response = google_images_download.googleimagesdownload()

    def test_similar_images_includes_params_in_url(self):
        """Test that params are included in URL when similar_images is used"""
        # Build arguments with similar_images and additional filters
        arguments = {
            'similar_images': 'https://example.com/image.jpg',
            'size': 'large',
            'format': 'jpg',
            'color': 'red',
            'language': None,
            'time_range': None,
            'exact_size': None,
            'color_type': None,
            'usage_rights': None,
            'type': None,
            'time': None,
            'aspect_ratio': None
        }

        # Build the params string
        params = self.response.build_url_parameters(arguments)

        # Build the search URL with similar_images
        url = self.response.build_search_url(
            search_term='',
            params=params,
            url=None,
            similar_images=arguments['similar_images'],
            specific_site=None,
            safe_search=False
        )

        # Verify that params are included in the URL
        # The params should contain size, format, and color filters
        self.assertIn('&tbs=', url, "URL should contain &tbs= parameter string")
        self.assertIn('isz:l', url, "URL should contain size:large parameter (isz:l)")
        self.assertIn('ift:jpg', url, "URL should contain format:jpg parameter (ift:jpg)")
        self.assertIn('ic:specific,isc:red', url, "URL should contain color:red parameter")

    def test_similar_images_with_size_filter(self):
        """Test that size filter works with similar_images"""
        arguments = {
            'similar_images': 'https://example.com/image.jpg',
            'size': 'medium',
            'language': None,
            'time_range': None,
            'exact_size': None,
            'color': None,
            'color_type': None,
            'usage_rights': None,
            'type': None,
            'time': None,
            'aspect_ratio': None,
            'format': None
        }

        params = self.response.build_url_parameters(arguments)
        url = self.response.build_search_url(
            search_term='',
            params=params,
            url=None,
            similar_images=arguments['similar_images'],
            specific_site=None,
            safe_search=False
        )

        self.assertIn('isz:m', url, "URL should contain size:medium parameter (isz:m)")

    def test_similar_images_with_format_filter(self):
        """Test that format filter works with similar_images"""
        arguments = {
            'similar_images': 'https://example.com/image.jpg',
            'format': 'png',
            'language': None,
            'time_range': None,
            'exact_size': None,
            'color': None,
            'color_type': None,
            'usage_rights': None,
            'size': None,
            'type': None,
            'time': None,
            'aspect_ratio': None
        }

        params = self.response.build_url_parameters(arguments)
        url = self.response.build_search_url(
            search_term='',
            params=params,
            url=None,
            similar_images=arguments['similar_images'],
            specific_site=None,
            safe_search=False
        )

        self.assertIn('ift:png', url, "URL should contain format:png parameter (ift:png)")

    def test_similar_images_with_multiple_filters(self):
        """Test that multiple filters work together with similar_images"""
        arguments = {
            'similar_images': 'https://example.com/image.jpg',
            'size': 'large',
            'format': 'jpg',
            'color': 'blue',
            'type': 'photo',
            'aspect_ratio': 'wide',
            'language': None,
            'time_range': None,
            'exact_size': None,
            'color_type': None,
            'usage_rights': None,
            'time': None
        }

        params = self.response.build_url_parameters(arguments)
        url = self.response.build_search_url(
            search_term='',
            params=params,
            url=None,
            similar_images=arguments['similar_images'],
            specific_site=None,
            safe_search=False
        )

        # Check all filters are present
        self.assertIn('isz:l', url, "URL should contain size:large")
        self.assertIn('ift:jpg', url, "URL should contain format:jpg")
        self.assertIn('ic:specific,isc:blue', url, "URL should contain color:blue")
        self.assertIn('itp:photo', url, "URL should contain type:photo")
        self.assertIn('iar:w', url, "URL should contain aspect_ratio:wide")


if __name__ == '__main__':
    unittest.main()
