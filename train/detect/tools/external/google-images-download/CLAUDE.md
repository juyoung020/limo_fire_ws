# google-images-download

## Repository Details
- **Name**: google-images-download
- **Owner**: hardikvasa
- **URL**: https://github.com/hardikvasa/google-images-download
- **Purpose**: Python script for searching and downloading hundreds of Google images to local disk
- **Documentation**: https://google-images-download.readthedocs.io

## Recent Major Updates
- **May 17, 2026**: Fixed filename sanitization issues
  - Added sanitize_filename() method to handle illegal characters in filenames
  - Search keywords with slashes, quotes, and other illegal characters now work properly
  - Prevents Windows/Unix filesystem errors when using extract_metadata or creating directories
  - Fixes issues #170 and #217

- **May 16, 2026**: Fixed compatibility with modern Google Images and Selenium 4.x
  - Updated to Selenium 4.x API (find_element with By locators, Service class)
  - Added webdriver-manager for automatic chromedriver installation
  - Rewrote image extraction for new Google Images HTML/JS structure
  - Added anti-detection measures
  - All existing features remain backward compatible

## Issue Management

### Common Issue Patterns
1. **Download failures** ("0 images downloaded", "can't download anything")
   - Usually caused by Google Images structure changes
   - Fixed in latest version (May 2026)

2. **TypeError: 'NoneType' object is not subscriptable**
   - Was caused by outdated Google Images parsing
   - Fixed in latest version

3. **Selenium/Chromedriver errors**
   - Fixed with Selenium 4.x update and webdriver-manager

4. **Filename/directory errors with special characters** (slashes, quotes, etc.)
   - Keywords with illegal characters (/, \, :, ", <, >, |, ?, *) causing filesystem errors
   - Fixed with automatic filename sanitization (May 2026)

### Issue Review Checklist
- Check if issue describes a bug fixed in recent commits
- Look at commit messages and git log to see if already addressed
- Verify issue date - older issues likely fixed by recent updates
- Check for duplicates before responding
- Close spam/invalid issues immediately

### Closing Issues - Tone Guidelines
- Keep brief and human
- For fixed issues: Reply in natural human tone explaining the issue is resolved and briefly describe what we did to fix it
  - Example: "This is now fixed. The similar_images feature now properly includes filter parameters like size and format in the search URL."
- For duplicates: "Closing as duplicate of #XXX"
- For spam: "Closing as spam" or "Closing as not a valid issue"

## PR Management
- Review for backward compatibility
- Test with various search terms and filters
- Check Selenium compatibility
- Verify documentation updates if API changes
- Run against different Python versions if possible

## Testing Best Practices
- Test basic download: `googleimagesdownload -k "cats" -l 5`
- Test with filters: format, color, size, time, usage_rights
- Test multiple keywords
- Test prefix and suffix options
- Test metadata extraction
- Verify chromedriver auto-installation works

## Code Standards
- Maintain Python 2/3 compatibility where possible
- Follow existing code style
- Keep all existing features backward compatible
- Update documentation for new features
- Add clear error messages

## Important Notes
- This tool is for educational purposes only
- Users must respect image copyrights
- Google Images structure changes frequently - be ready to update parsing logic
- Selenium is now required (Google Images needs JavaScript)
