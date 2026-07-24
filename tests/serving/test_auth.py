from __future__ import annotations

import pytest

from bharat.serving.auth import ApiKeyAuthenticator, AuthConfig, AuthResult


class TestAuthConfig:
    def test_valid(self) -> None:
        config = AuthConfig(valid_api_keys=("key-1", "key-2"))
        assert len(config.valid_api_keys) == 2

    def test_empty_keys_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one valid API key is required"):
            AuthConfig(valid_api_keys=())


class TestAuthResult:
    def test_success(self) -> None:
        result = AuthResult(ok=True, client_id="test-user")
        assert result.ok
        assert result.error is None

    def test_failure_with_error(self) -> None:
        result = AuthResult(ok=False, error="Invalid key")
        assert not result.ok
        assert result.error == "Invalid key"

    def test_failure_without_error_raises(self) -> None:
        with pytest.raises(ValueError, match="Auth failure must include an error message"):
            AuthResult(ok=False)

    def test_success_with_error_raises(self) -> None:
        with pytest.raises(ValueError, match="Auth success must not include an error message"):
            AuthResult(ok=True, error="Should not happen")


class TestApiKeyAuthenticator:
    def _make_auth(self) -> ApiKeyAuthenticator:
        config = AuthConfig(valid_api_keys=("valid-key-001", "valid-key-002"))
        return ApiKeyAuthenticator(config)

    def test_valid_api_key_accepted(self) -> None:
        auth = self._make_auth()
        result = auth.authenticate("valid-key-001")
        assert result.ok
        assert result.client_id == "valid-ke"

    def test_invalid_api_key_rejected(self) -> None:
        auth = self._make_auth()
        result = auth.authenticate("invalid-key")
        assert not result.ok
        assert result.error == "Invalid API key"

    def test_missing_api_key_rejected(self) -> None:
        auth = self._make_auth()
        result = auth.authenticate(None)
        assert not result.ok
        assert result.error == "Missing API key"

    def test_empty_string_api_key_rejected(self) -> None:
        auth = self._make_auth()
        result = auth.authenticate("")
        assert not result.ok
        assert result.error == "Invalid API key"

    def test_multiple_valid_keys(self) -> None:
        auth = self._make_auth()
        assert auth.authenticate("valid-key-001").ok
        assert auth.authenticate("valid-key-002").ok

    def test_deterministic_client_id(self) -> None:
        auth = self._make_auth()
        result1 = auth.authenticate("valid-key-001")
        result2 = auth.authenticate("valid-key-001")
        assert result1.client_id == result2.client_id
