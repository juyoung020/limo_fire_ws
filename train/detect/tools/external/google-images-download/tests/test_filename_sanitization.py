"""
Test for issue #217 and #170: Filenames with illegal characters cause errors
This test verifies that search terms with slashes, quotes, and other illegal
characters are properly sanitized before being used as filenames or directory names.
"""

from google_images_download import google_images_download
import unittest


class TestFilenameSanitization(unittest.TestCase):

    def setUp(self):
        self.response = google_images_download.googleimagesdownload()

    def test_sanitize_filename_removes_slashes(self):
        """Test that slashes are removed from filenames"""
        filename = "SOME ITEM/demo"
        sanitized = self.response.sanitize_filename(filename)
        self.assertNotIn('/', sanitized)
        self.assertEqual(sanitized, "SOME ITEM_demo")

    def test_sanitize_filename_removes_quotes(self):
        """Test that quotes are removed from filenames"""
        filename = '"siberian husky"'
        sanitized = self.response.sanitize_filename(filename)
        self.assertNotIn('"', sanitized)
        self.assertEqual(sanitized, "_siberian husky_")

    def test_sanitize_filename_removes_all_illegal_chars(self):
        """Test that all illegal Windows/Unix characters are removed"""
        filename = 'test<>:"|\\?*file'
        sanitized = self.response.sanitize_filename(filename)
        illegal_chars = '<>:"/\\|?*'
        for char in illegal_chars:
            self.assertNotIn(char, sanitized)
        self.assertEqual(sanitized, "test________file")

    def test_sanitize_filename_strips_trailing_dots_and_spaces(self):
        """Test that trailing dots and spaces are removed (Windows issue)"""
        filename = "test file. "
        sanitized = self.response.sanitize_filename(filename)
        self.assertEqual(sanitized, "test file")

    def test_sanitize_filename_normal_text_unchanged(self):
        """Test that normal filenames remain unchanged"""
        filename = "normal_filename-123"
        sanitized = self.response.sanitize_filename(filename)
        self.assertEqual(sanitized, filename)


if __name__ == '__main__':
    unittest.main()
