from __future__ import annotations

import re

_VERSION_RE = re.compile(
    r"^(?P<major>0|[1-9]\d*)"
    r"\.(?P<minor>0|[1-9]\d*)"
    r"\.(?P<patch>0|[1-9]\d*)"
    r"(?P<prerelease>-(alpha|beta|rc)\d*)?"
    r"$"
)


class Version:
    """Strict semantic version with PEP 440-compatible ordering."""

    def __init__(self, major: int, minor: int, patch: int, prerelease: str | None = None) -> None:
        self._major = major
        self._minor = minor
        self._patch = patch
        self._prerelease = prerelease

    @classmethod
    def parse(cls, s: str) -> Version:
        m = _VERSION_RE.match(s)
        if not m:
            raise ValueError(f"invalid version: '{s}'")
        return cls(
            major=int(m.group("major")),
            minor=int(m.group("minor")),
            patch=int(m.group("patch")),
            prerelease=m.group("prerelease"),
        )

    def __str__(self) -> str:
        s = f"{self._major}.{self._minor}.{self._patch}"
        if self._prerelease:
            s += self._prerelease
        return s

    def __repr__(self) -> str:
        return f"Version('{self}')"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return (
            self._major == other._major
            and self._minor == other._minor
            and self._patch == other._patch
            and self._prerelease == other._prerelease
        )

    def __hash__(self) -> int:
        return hash((self._major, self._minor, self._patch, self._prerelease))

    def __lt__(self, other: Version) -> bool:
        if self._major != other._major:
            return self._major < other._major
        if self._minor != other._minor:
            return self._minor < other._minor
        if self._patch != other._patch:
            return self._patch < other._patch
        if self._prerelease and not other._prerelease:
            return True
        if not self._prerelease and other._prerelease:
            return False
        return (self._prerelease or "") < (other._prerelease or "")

    def __le__(self, other: Version) -> bool:
        return self < other or self == other

    def __gt__(self, other: Version) -> bool:
        return not (self <= other)

    def __ge__(self, other: Version) -> bool:
        return not (self < other)

    def normalized(self) -> str:
        return str(self)
