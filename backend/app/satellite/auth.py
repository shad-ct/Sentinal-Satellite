"""
Copernicus Data Space Ecosystem (CDSE) authentication.

Manages OAuth2 client-credentials token acquisition and caching.
Tokens are refreshed automatically before expiry.
Credentials are NEVER written to logs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import requests

from backend.app.utils.config import get_settings
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)

# Refresh token this many seconds before it actually expires
_REFRESH_BUFFER_SECONDS = 60


@dataclass
class Token:
    """An OAuth2 bearer token with expiry tracking."""

    access_token: str
    expires_at: float = field(default_factory=lambda: time.time() + 3600)

    def is_expired(self) -> bool:
        """Return True if the token needs refresh."""
        return time.time() >= (self.expires_at - _REFRESH_BUFFER_SECONDS)

    def as_header(self) -> dict[str, str]:
        """Return the Authorization header dict."""
        return {"Authorization": f"Bearer {self.access_token}"}


class CDSEAuthenticator:
    """
    Handles OAuth2 machine-to-machine authentication with the
    Copernicus Data Space Ecosystem (CDSE) identity server.

    Token caching ensures we don't request a new token on every API call.
    """

    def __init__(self) -> None:
        self._token: Optional[Token] = None
        self._settings = get_settings()

    def get_token(self) -> Token:
        """
        Return a valid bearer token, refreshing if necessary.

        Raises:
            RuntimeError: If credentials are missing or auth fails.
        """
        if self._token and not self._token.is_expired():
            return self._token

        logger.info("Requesting new Copernicus authentication token...")
        self._token = self._fetch_token()
        logger.info("Authentication token acquired (valid for ~1 h).")
        return self._token

    def _fetch_token(self) -> Token:
        """
        Request a new token from the CDSE OAuth2 endpoint.

        Returns:
            A Token instance.

        Raises:
            RuntimeError: On authentication failure.
        """
        settings = self._settings
        settings.require_copernicus_credentials()

        try:
            response = requests.post(
                settings.cdse_token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": settings.copernicus_client_id,
                    "client_secret": settings.copernicus_client_secret,
                },
                timeout=30,
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                "Cannot connect to Copernicus authentication server. "
                "Check your network connection."
            ) from exc
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response else "?"
            raise RuntimeError(
                f"Copernicus authentication failed (HTTP {status}). "
                "Verify COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET in your .env file."
            ) from exc

        payload = response.json()
        access_token = payload.get("access_token")
        if not access_token:
            raise RuntimeError(
                "Copernicus auth response missing 'access_token'. "
                "Unexpected response format."
            )

        expires_in = int(payload.get("expires_in", 3600))
        return Token(
            access_token=access_token,
            expires_at=time.time() + expires_in,
        )

    def get_auth_headers(self) -> dict[str, str]:
        """
        Return the Authorization header dict for API requests.

        Returns:
            Dict with 'Authorization' key.
        """
        return self.get_token().as_header()


# Module-level singleton for convenience
_authenticator: Optional[CDSEAuthenticator] = None


def get_authenticator() -> CDSEAuthenticator:
    """Return the shared CDSEAuthenticator singleton."""
    global _authenticator
    if _authenticator is None:
        _authenticator = CDSEAuthenticator()
    return _authenticator
