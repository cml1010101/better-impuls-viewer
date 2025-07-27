"""
Google OAuth2 authentication for Better Impuls Viewer.
Replaces service account authentication with user OAuth flow.
"""

import os
import json
import requests
from typing import Dict, Any, Optional, Tuple
from urllib.parse import urlencode, parse_qs, urlparse
import hashlib
import time
from dataclasses import dataclass

from credentials_manager import get_credentials_manager


@dataclass
class GoogleOAuthConfig:
    """Google OAuth2 configuration."""
    # For development/local testing, you would set these
    # In production, these should be environment variables or part of app config
    client_id: str = "your-client-id.apps.googleusercontent.com"
    client_secret: str = "your-client-secret" 
    redirect_uri: str = "http://localhost:8000/oauth/google/callback"
    scopes: list = None
    
    def __post_init__(self):
        if self.scopes is None:
            self.scopes = [
                'https://www.googleapis.com/auth/spreadsheets.readonly',
                'https://www.googleapis.com/auth/userinfo.email'
            ]


class GoogleOAuthManager:
    """Manages Google OAuth2 authentication flow."""
    
    def __init__(self):
        """Initialize OAuth manager."""
        self.config = GoogleOAuthConfig()
        self.credentials_manager = get_credentials_manager()
        self._setup_oauth_config()
    
    def _setup_oauth_config(self):
        """Setup OAuth configuration from environment or defaults."""
        # Try to get OAuth config from environment variables
        env_client_id = os.getenv('GOOGLE_OAUTH_CLIENT_ID')
        env_client_secret = os.getenv('GOOGLE_OAUTH_CLIENT_SECRET')
        env_redirect_uri = os.getenv('GOOGLE_OAUTH_REDIRECT_URI')
        
        if env_client_id:
            self.config.client_id = env_client_id
        if env_client_secret:
            self.config.client_secret = env_client_secret
        if env_redirect_uri:
            self.config.redirect_uri = env_redirect_uri
    
    def get_authorization_url(self, state: Optional[str] = None) -> str:
        """
        Generate Google OAuth2 authorization URL.
        
        Args:
            state: Optional state parameter for CSRF protection
            
        Returns:
            Authorization URL for user to visit
        """
        if not state:
            state = hashlib.sha256(f"{time.time()}".encode()).hexdigest()[:16]
        
        params = {
            'client_id': self.config.client_id,
            'redirect_uri': self.config.redirect_uri,
            'scope': ' '.join(self.config.scopes),
            'response_type': 'code',
            'access_type': 'offline',  # To get refresh token
            'prompt': 'consent',  # Force consent to ensure refresh token
            'state': state
        }
        
        base_url = 'https://accounts.google.com/o/oauth2/v2/auth'
        return f"{base_url}?{urlencode(params)}"
    
    def exchange_code_for_tokens(self, authorization_code: str) -> Dict[str, Any]:
        """
        Exchange authorization code for access and refresh tokens.
        
        Args:
            authorization_code: Code received from OAuth callback
            
        Returns:
            Token response with access_token, refresh_token, etc.
        """
        token_url = 'https://oauth2.googleapis.com/token'
        
        data = {
            'client_id': self.config.client_id,
            'client_secret': self.config.client_secret,
            'code': authorization_code,
            'grant_type': 'authorization_code',
            'redirect_uri': self.config.redirect_uri
        }
        
        response = requests.post(token_url, data=data)
        response.raise_for_status()
        
        token_data = response.json()
        
        # Store tokens in credentials manager
        access_token = token_data.get('access_token')
        refresh_token = token_data.get('refresh_token')
        
        if access_token:
            self.credentials_manager.set_google_oauth_tokens(access_token, refresh_token)
        
        return token_data
    
    def refresh_access_token(self) -> Optional[str]:
        """
        Refresh the access token using the stored refresh token.
        
        Returns:
            New access token if successful, None otherwise
        """
        refresh_token = self.credentials_manager.get_google_refresh_token()
        if not refresh_token:
            return None
        
        token_url = 'https://oauth2.googleapis.com/token'
        
        data = {
            'client_id': self.config.client_id,
            'client_secret': self.config.client_secret,
            'refresh_token': refresh_token,
            'grant_type': 'refresh_token'
        }
        
        try:
            response = requests.post(token_url, data=data)
            response.raise_for_status()
            
            token_data = response.json()
            new_access_token = token_data.get('access_token')
            
            if new_access_token:
                # Update the stored access token (keep the same refresh token)
                self.credentials_manager.set_google_oauth_tokens(new_access_token)
                return new_access_token
                
        except Exception as e:
            print(f"Error refreshing access token: {e}")
            return None
        
        return None
    
    def get_valid_access_token(self) -> Optional[str]:
        """
        Get a valid access token, refreshing if necessary.
        
        Returns:
            Valid access token or None if authentication failed
        """
        access_token = self.credentials_manager.get_google_oauth_token()
        
        if not access_token:
            return None
        
        # Test if the current token is valid
        if self._is_token_valid(access_token):
            return access_token
        
        # Try to refresh the token
        new_token = self.refresh_access_token()
        return new_token
    
    def _is_token_valid(self, access_token: str) -> bool:
        """
        Check if an access token is still valid.
        
        Args:
            access_token: Token to validate
            
        Returns:
            True if token is valid, False otherwise
        """
        try:
            # Test token by making a simple API call
            headers = {'Authorization': f'Bearer {access_token}'}
            response = requests.get(
                'https://www.googleapis.com/oauth2/v1/tokeninfo',
                headers=headers,
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def get_user_info(self, access_token: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get user information using the access token.
        
        Args:
            access_token: Optional access token. If not provided, will try to get valid token.
            
        Returns:
            User information dict or None if failed
        """
        if not access_token:
            access_token = self.get_valid_access_token()
        
        if not access_token:
            return None
        
        try:
            headers = {'Authorization': f'Bearer {access_token}'}
            response = requests.get(
                'https://www.googleapis.com/oauth2/v2/userinfo',
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting user info: {e}")
            return None
    
    def revoke_authentication(self) -> bool:
        """
        Revoke the stored authentication tokens.
        
        Returns:
            True if successful, False otherwise
        """
        access_token = self.credentials_manager.get_google_oauth_token()
        
        success = True
        if access_token:
            try:
                # Revoke the token with Google
                revoke_url = 'https://oauth2.googleapis.com/revoke'
                params = {'token': access_token}
                response = requests.post(revoke_url, params=params, timeout=10)
                # Google returns 200 for success or if token was already invalid
                success = response.status_code == 200
            except Exception as e:
                print(f"Error revoking token with Google: {e}")
                success = False
        
        # Clear stored credentials regardless of revocation result
        self.credentials_manager.clear_google_credentials()
        return success
    
    def is_authenticated(self) -> bool:
        """
        Check if user is currently authenticated with Google.
        
        Returns:
            True if authenticated, False otherwise
        """
        return self.get_valid_access_token() is not None
    
    def get_oauth_status(self) -> Dict[str, Any]:
        """
        Get current OAuth authentication status.
        
        Returns:
            Dictionary with authentication status information
        """
        access_token = self.credentials_manager.get_google_oauth_token()
        refresh_token = self.credentials_manager.get_google_refresh_token()
        
        status = {
            'authenticated': False,
            'has_access_token': access_token is not None,
            'has_refresh_token': refresh_token is not None,
            'user_info': None,
            'token_valid': False
        }
        
        if access_token:
            status['token_valid'] = self._is_token_valid(access_token)
            status['authenticated'] = status['token_valid']
            
            if status['token_valid']:
                status['user_info'] = self.get_user_info(access_token)
        
        return status


# Global OAuth manager instance
_oauth_manager = None

def get_oauth_manager() -> GoogleOAuthManager:
    """Get the global OAuth manager instance."""
    global _oauth_manager
    if _oauth_manager is None:
        _oauth_manager = GoogleOAuthManager()
    return _oauth_manager