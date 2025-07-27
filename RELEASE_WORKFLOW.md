# Release Workflow Documentation

This repository includes an automated GitHub Actions workflow that builds and distributes the Better Impuls Viewer Electron application when a new release is created.

## How It Works

The release workflow (`.github/workflows/release.yml`) automatically:

1. **Triggers on Release Creation**: Runs whenever a new GitHub release is created
2. **Sets Up Build Environment**: Installs Node.js, Python, and system dependencies
3. **Builds the Application**: Uses the existing `scripts/build.sh` script to create the Electron package
4. **Generates Checksums**: Creates SHA256 checksums for integrity verification
5. **Uploads Assets**: Attaches the AppImage and checksum files to the GitHub release

## Creating a Release

To trigger automatic distribution:

1. **Create a new tag**:
   ```bash
   git tag v1.0.1
   git push origin v1.0.1
   ```

2. **Create a GitHub release**:
   - Go to the repository's "Releases" page
   - Click "Create a new release"
   - Select the tag you just created
   - Add release notes describing changes
   - Click "Publish release"

3. **Automatic Build**: The workflow will automatically start building and will attach the distributable files to the release

## Manual Testing

The workflow can also be triggered manually for testing purposes:

1. Go to the "Actions" tab in the GitHub repository
2. Select "Build and Release Distribution" workflow
3. Click "Run workflow"
4. Optionally specify a tag name for the build
5. The built artifacts will be available as workflow artifacts (not attached to a release)

## Generated Assets

Each release will include:

- **AppImage File**: `better-impuls-viewer-{tag}.AppImage` - The main distributable application
- **Checksum File**: `better-impuls-viewer-{tag}.sha256` - SHA256 hash for integrity verification

## Verifying Downloads

Users can verify the integrity of downloaded files:

```bash
# Download both the AppImage and checksum file
wget https://github.com/{org}/{repo}/releases/download/{tag}/better-impuls-viewer-{tag}.AppImage
wget https://github.com/{org}/{repo}/releases/download/{tag}/better-impuls-viewer-{tag}.sha256

# Verify the checksum
sha256sum -c better-impuls-viewer-{tag}.sha256
```

## Workflow Configuration

The workflow includes:

- **Ubuntu latest** runner for Linux AppImage builds
- **Node.js 18** for Electron and frontend building
- **Python 3.9** for backend dependencies
- **System dependencies** required for Electron packaging
- **Error handling** and artifact validation
- **Conditional uploads** (release assets vs workflow artifacts)

## Troubleshooting

If the workflow fails:

1. **Check the build logs** in the Actions tab
2. **Verify dependencies** in `requirements.txt` and `package.json`
3. **Test locally** using `./scripts/build.sh`
4. **Check disk space** - builds can be large (100MB+)

## Future Enhancements

Potential improvements:

- **Multi-platform builds** (Windows, macOS)
- **Automated testing** before release
- **Code signing** for trusted distributions
- **Release notes automation** from commit messages