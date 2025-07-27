# Google OAuth Setup Guide

## Problem Resolved

**Previous Issue:** "OAuth client was not found" error when trying to authenticate with Google Sheets.

**Root Cause:** The application was using placeholder OAuth credentials instead of real Google Cloud Console credentials.

**Solution:** The app now allows users to configure their own Google OAuth credentials through the Settings UI.

## How to Set Up Google OAuth

### Step 1: Create Google Cloud Console Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Sign in with your Google account
3. Create a new project or select an existing one

### Step 2: Enable Required APIs

1. Go to "APIs & Services" → "Library"
2. Search for and enable:
   - **Google Sheets API**
   - **Google Drive API** (optional, for enhanced access)

### Step 3: Create OAuth Credentials

1. Go to "APIs & Services" → "Credentials"
2. Click "Create Credentials" → "OAuth client ID"
3. If prompted, configure the OAuth consent screen:
   - Choose "External" for personal use
   - Fill in the required application information
   - Add your email to test users if needed

### Step 4: Configure OAuth Client

1. Choose "Web application" as the application type
2. Add authorized redirect URI: `http://localhost:8000/oauth/google/callback`
3. Click "Create"
4. **Important:** Copy the Client ID and Client Secret

### Step 5: Configure in Better Impuls Viewer

1. Open the application
2. Click the Settings button (⚙️)
3. In the Google Sheets Configuration section:
   - If OAuth is not configured, you'll see setup instructions
   - Click "Show Setup Instructions"
   - Paste your Client ID and Client Secret
   - Click "Configure OAuth"

### Step 6: Authenticate

1. Set your Google Sheets URL in Settings
2. Click "Authenticate with Google"
3. Complete the OAuth flow in the popup window
4. You should now be authenticated successfully!

## Security Notes

- Your OAuth credentials are stored locally and encrypted
- The Client Secret is stored securely and never transmitted to external servers
- You can clear the OAuth configuration at any time from Settings

## Troubleshooting

**"Invalid client ID format"**: Make sure your Client ID ends with `.apps.googleusercontent.com`

**"Client secret too short"**: Ensure you copied the complete Client Secret from Google Cloud Console

**Authentication popup blocked**: Check your browser's popup blocker settings

**Still getting "OAuth client not found"**: Double-check that you've correctly copied both the Client ID and Client Secret, and that your Google Cloud project has the required APIs enabled.

## For Developers

This fix replaces the hardcoded placeholder OAuth credentials with a user-configurable system:

- **Backend**: Extended `credentials_manager.py` to handle OAuth client credentials
- **OAuth Manager**: Updated `google_oauth.py` to use user-provided credentials with proper validation
- **Frontend**: Enhanced Settings UI to guide users through OAuth setup
- **API**: Added endpoints for OAuth configuration and setup instructions

The implementation ensures users have full control over their OAuth setup while providing clear guidance for the configuration process.