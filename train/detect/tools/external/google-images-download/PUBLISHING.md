# Publishing to PyPI

This repository is configured to automatically publish to PyPI when you create a new release tag.

## One-Time Setup

### 1. Create PyPI API Token

1. Go to https://pypi.org/manage/account/token/
2. Create a new API token with name "github-actions"
3. Scope: "Entire account" or specific to "google-images-download" project
4. Copy the token (starts with `pypi-`)

### 2. Add Token to GitHub Secrets

1. Go to https://github.com/hardikvasa/google-images-download/settings/secrets/actions
2. Click "New repository secret"
3. Name: `PYPI_API_TOKEN`
4. Value: Paste your PyPI token
5. Click "Add secret"

## How to Publish a New Version

### Automatic Publishing (Recommended)

1. Update version in `setup.py` and `pyproject.toml`
2. Commit and push changes
3. Create and push a git tag:
   ```bash
   git tag v3.0.0
   git push origin v3.0.0
   ```
4. GitHub Actions will automatically build and publish to PyPI

### Manual Publishing (Fallback)

If you need to publish manually:

```bash
# Install build tools
pip install build twine

# Build the package
python -m build

# Upload to PyPI
twine upload dist/*
```

You'll be prompted for your PyPI username and password/token.

## Version Numbering

Follow semantic versioning:
- Major (X.0.0): Breaking changes or major rewrites
- Minor (3.X.0): New features, backward compatible
- Patch (3.0.X): Bug fixes

Current version: 3.0.0 (major update with Selenium 4.x and modern Google Images support)
