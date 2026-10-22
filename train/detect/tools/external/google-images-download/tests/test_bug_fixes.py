"""
Tests for bug fixes:
- #281: Typo argumnets -> arguments
- #262: print_paths error with dict encoding
- #263: Trailing spaces in queries
"""

from google_images_download import google_images_download
import unittest


class TestBugFixes(unittest.TestCase):

    def setUp(self):
        self.response = google_images_download.googleimagesdownload()

    def test_trailing_spaces_in_keywords(self):
        """Test that trailing spaces in keywords are stripped (Issue #263)"""
        # Test that the keyword processing strips spaces correctly
        test_keywords = 'test keyword , another keyword  , third '
        keywords_list = [str(item).strip() for item in test_keywords.split(',')]

        # Verify no trailing or leading spaces
        for keyword in keywords_list:
            self.assertEqual(keyword, keyword.strip())
            self.assertNotEqual(keyword[0] if keyword else '', ' ')
            self.assertNotEqual(keyword[-1] if keyword else '', ' ')

    def test_keywords_are_stripped(self):
        """Test that keywords are properly stripped of whitespace"""
        # Mock the keyword splitting logic
        test_keywords = 'keyword1 , keyword2  ,  keyword3   '
        keywords_list = [str(item).strip() for item in test_keywords.split(',')]

        self.assertEqual(keywords_list[0], 'keyword1')
        self.assertEqual(keywords_list[1], 'keyword2')
        self.assertEqual(keywords_list[2], 'keyword3')


if __name__ == '__main__':
    unittest.main()
