from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AuthConfig:
    valid_api_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.valid_api_keys:
            raise ValueError("At least one valid API key is required")


@dataclass(frozen=True)
class AuthResult:
    ok: bool
    error: str | None = None
    client_id: str = ""

    def __post_init__(self) -> None:
        if not self.ok and not self.error:
            raise ValueError("Auth failure must include an error message")
        if self.ok and self.error:
            raise ValueError("Auth success must not include an error message")


class ApiKeyAuthenticator:
    def __init__(self, config: AuthConfig) -> None:
        self._valid_keys = set(config.valid_api_keys)

    def authenticate(self, api_key: str | None) -> AuthResult:
        if api_key is None:
            return AuthResult(ok=False, error="Missing API key")
        if api_key not in self._valid_keys:
            return AuthResult(ok=False, error="Invalid API key")
        client_id = api_key[:8]
        return AuthResult(ok=True, client_id=client_id)
