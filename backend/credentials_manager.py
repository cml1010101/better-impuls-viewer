"""
Credentials management system for Better Impuls Viewer.
Handles secure storage and retrieval of user credentials without .env files.
"""

import os
import json
import hashlib
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppCredentials:
    """Application credentials data structure."""
    google_sheets_url: Optional[str] = None
    google_oauth_token: Optional[str] = None
    google_refresh_token: Optional[str] = None
    google_oauth_client_id: Optional[str] = None
    google_oauth_client_secret: Optional[str] = None
    sed_url: Optional[str] = None
    sed_username: Optional[str] = None
    sed_password: Optional[str] = None
    data_folder_path: Optional[str] = None


class CredentialsManager:
    """Secure credentials management for the application."""
    
    def __init__(self):
        """Initialize credentials manager with secure storage location."""
        # Use different storage locations for different environments
        if os.environ.get('DATA_FOLDER'):
            # Electron environment - use app data folder
            self.storage_dir = Path(os.environ.get('DATA_FOLDER')) / 'config'
        else:
            # Development environment - use local config
            self.storage_dir = Path.home() / '.impuls-viewer'
        
        self.storage_dir.mkdir(exist_ok=True)
        self.credentials_file = self.storage_dir / 'credentials.json'
        self._credentials = AppCredentials()
        self._load_credentials()
    
    def _get_file_hash(self, content: str) -> str:
        """Generate hash for basic content validation."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _load_credentials(self) -> None:
        """Load credentials from secure storage."""
        if not self.credentials_file.exists():
            return
        
        try:
            with open(self.credentials_file, 'r') as f:
                data = json.load(f)
            
            # Validate file integrity (basic check)
            content = json.dumps(data.get('credentials', {}), sort_keys=True)
            stored_hash = data.get('hash', '')
            expected_hash = self._get_file_hash(content)
            
            if stored_hash != expected_hash:
                print("Warning: Credentials file may have been modified")
                return
            
            creds_data = data.get('credentials', {})
            self._credentials = AppCredentials(**creds_data)
            
        except Exception as e:
            print(f"Error loading credentials: {e}")
            self._credentials = AppCredentials()
    
    def _save_credentials(self) -> None:
        """Save credentials to secure storage."""
        try:
            creds_dict = {
                'google_sheets_url': self._credentials.google_sheets_url,
                'google_oauth_token': self._credentials.google_oauth_token,
                'google_refresh_token': self._credentials.google_refresh_token,
                'google_oauth_client_id': self._credentials.google_oauth_client_id,
                'google_oauth_client_secret': self._credentials.google_oauth_client_secret,
                'sed_url': self._credentials.sed_url,
                'sed_username': self._credentials.sed_username,
                'sed_password': self._credentials.sed_password,
                'data_folder_path': self._credentials.data_folder_path
            }
            
            # Remove None values
            creds_dict = {k: v for k, v in creds_dict.items() if v is not None}
            
            content = json.dumps(creds_dict, sort_keys=True)
            file_hash = self._get_file_hash(content)
            
            data = {
                'credentials': creds_dict,
                'hash': file_hash,
                'version': '1.0'
            }
            
            # Atomic write
            temp_file = self.credentials_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            temp_file.replace(self.credentials_file)
            
            # Set restrictive permissions (owner read/write only)
            os.chmod(self.credentials_file, 0o600)
            
        except Exception as e:
            print(f"Error saving credentials: {e}")
    
    def get_google_sheets_url(self) -> Optional[str]:
        """Get Google Sheets URL."""
        return self._credentials.google_sheets_url
    
    def set_google_sheets_url(self, url: str) -> None:
        """Set Google Sheets URL."""
        self._credentials.google_sheets_url = url
        self._save_credentials()
    
    def get_google_oauth_token(self) -> Optional[str]:
        """Get Google OAuth access token."""
        return self._credentials.google_oauth_token
    
    def set_google_oauth_tokens(self, access_token: str, refresh_token: Optional[str] = None) -> None:
        """Set Google OAuth tokens."""
        self._credentials.google_oauth_token = access_token
        if refresh_token:
            self._credentials.google_refresh_token = refresh_token
        self._save_credentials()
    
    def get_google_refresh_token(self) -> Optional[str]:
        """Get Google OAuth refresh token."""
        return self._credentials.google_refresh_token
    
    def get_google_oauth_client_credentials(self) -> Dict[str, Optional[str]]:
        """Get Google OAuth client credentials."""
        return {
            'client_id': self._credentials.google_oauth_client_id,
            'client_secret': self._credentials.google_oauth_client_secret
        }
    
    def set_google_oauth_client_credentials(self, client_id: str, client_secret: str) -> None:
        """Set Google OAuth client credentials."""
        self._credentials.google_oauth_client_id = client_id
        self._credentials.google_oauth_client_secret = client_secret
        self._save_credentials()
    
    def has_google_oauth_client_credentials(self) -> bool:
        """Check if Google OAuth client credentials are configured."""
        return (self._credentials.google_oauth_client_id is not None and 
                self._credentials.google_oauth_client_secret is not None)
    
    def get_sed_credentials(self) -> Dict[str, Optional[str]]:
        """Get SED service credentials."""
        return {
            'url': self._credentials.sed_url,
            'username': self._credentials.sed_username,
            'password': self._credentials.sed_password
        }
    
    def set_sed_credentials(self, url: str, username: str, password: str) -> None:
        """Set SED service credentials."""
        self._credentials.sed_url = url
        self._credentials.sed_username = username
        self._credentials.sed_password = password
        self._save_credentials()
    
    def clear_google_credentials(self) -> None:
        """Clear Google authentication credentials."""
        self._credentials.google_oauth_token = None
        self._credentials.google_refresh_token = None
        self._save_credentials()
    
    def clear_google_oauth_client_credentials(self) -> None:
        """Clear Google OAuth client credentials."""
        self._credentials.google_oauth_client_id = None
        self._credentials.google_oauth_client_secret = None
        self._save_credentials()
    
    def clear_all_credentials(self) -> None:
        """Clear all stored credentials."""
        self._credentials = AppCredentials()
        if self.credentials_file.exists():
            self.credentials_file.unlink()
    
    def has_google_auth(self) -> bool:
        """Check if Google authentication is configured."""
        return (self._credentials.google_oauth_token is not None and 
                self._credentials.google_sheets_url is not None)
    
    def has_sed_credentials(self) -> bool:
        """Check if SED credentials are configured."""
        return (self._credentials.sed_url is not None and 
                self._credentials.sed_username is not None and 
                self._credentials.sed_password is not None)
    
    def get_credentials_status(self) -> Dict[str, bool]:
        """Get status of all credential types."""
        return {
            'google_sheets': self.has_google_auth(),
            'sed_service': self.has_sed_credentials(),
            'google_sheets_url_set': self._credentials.google_sheets_url is not None,
            'data_folder_configured': self._credentials.data_folder_path is not None,
            'google_oauth_client_configured': self.has_google_oauth_client_credentials()
        }
    
    def get_data_folder_path(self) -> Optional[str]:
        """Get configured data folder path."""
        return self._credentials.data_folder_path
    
    def set_data_folder_path(self, path: str) -> None:
        """Set data folder path."""
        # Validate the path exists
        if not os.path.exists(path):
            raise ValueError(f"Data folder path does not exist: {path}")
        if not os.path.isdir(path):
            raise ValueError(f"Data folder path is not a directory: {path}")
        
        self._credentials.data_folder_path = os.path.abspath(path)
        self._save_credentials()
    
    def clear_data_folder_path(self) -> None:
        """Clear configured data folder path."""
        self._credentials.data_folder_path = None
        self._save_credentials()


# Global credentials manager instance
_credentials_manager = None

def get_credentials_manager() -> CredentialsManager:
    """Get the global credentials manager instance."""
    global _credentials_manager
    if _credentials_manager is None:
        _credentials_manager = CredentialsManager()
    return _credentials_manager