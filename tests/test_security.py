"""Tests for cuecard.security."""

from __future__ import annotations

import os
import stat
import warnings
from pathlib import Path

import pytest

from cuecard.security import (
    ConfigError,
    ensure_directory,
    scrub_secrets,
    secure_open,
    validate_source_path,
)


class TestScrubSecrets:
    def test_stripe_key(self) -> None:
        text = "key is sk_live_abc123XYZ456789012345678"
        assert "[REDACTED]" in scrub_secrets(text)
        assert "sk_live_" not in scrub_secrets(text)

    def test_stripe_test_key(self) -> None:
        text = "key is sk_test_abc123XYZ456789012345678"
        assert "[REDACTED]" in scrub_secrets(text)
        assert "sk_test_" not in scrub_secrets(text)

    def test_aws_key(self) -> None:
        text = "aws key AKIAIOSFODNN7EXAMPLE"
        result = scrub_secrets(text)
        assert "AKIA" not in result
        assert "[REDACTED]" in result

    def test_github_pat(self) -> None:
        text = "token ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx1234"
        result = scrub_secrets(text)
        assert "ghp_" not in result
        assert "[REDACTED]" in result

    def test_connection_string(self) -> None:
        text = "DATABASE_URL=postgresql://admin:s3cret@db.host:5432/mydb"
        result = scrub_secrets(text)
        assert "s3cret" not in result
        assert "[REDACTED]" in result

    def test_bearer_token(self) -> None:
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig"
        result = scrub_secrets(text)
        assert "eyJ" not in result
        assert "Bearer [REDACTED]" in result

    def test_generic_secret_key(self) -> None:
        text = "MY_SECRET=supersecretvalue"
        result = scrub_secrets(text)
        assert "supersecretvalue" not in result
        assert "MY_SECRET=[REDACTED]" in result

    def test_generic_token(self) -> None:
        text = "AUTH_TOKEN=tok_12345"
        result = scrub_secrets(text)
        assert "tok_12345" not in result

    def test_generic_password(self) -> None:
        text = "DB_PASSWORD=hunter2"
        result = scrub_secrets(text)
        assert "hunter2" not in result

    def test_generic_api_key(self) -> None:
        text = "OPENAI_API_KEY=sk-proj-abc123"
        result = scrub_secrets(text)
        assert "sk-proj-abc123" not in result

    def test_no_secrets(self) -> None:
        text = "This is a normal line with no secrets"
        assert scrub_secrets(text) == text

    def test_multiple_secrets(self) -> None:
        text = "AKIAIOSFODNN7EXAMPLE and ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx1234"
        result = scrub_secrets(text)
        assert "AKIA" not in result
        assert "ghp_" not in result

    def test_anthropic_key(self) -> None:
        text = "key sk-ant-api03-abcdefghij1234567890"
        result = scrub_secrets(text)
        assert "sk-ant-" not in result
        assert "[REDACTED]" in result

    def test_openai_project_key(self) -> None:
        text = "key sk-proj-abcdefghij1234567890AB"
        result = scrub_secrets(text)
        assert "sk-proj-" not in result
        assert "[REDACTED]" in result

    def test_openai_legacy_key(self) -> None:
        text = "key sk-" + "a" * 40
        result = scrub_secrets(text)
        assert "sk-" + "a" * 40 not in result
        assert "[REDACTED]" in result

    def test_slack_token(self) -> None:
        text = "token xoxb-123456-789012-abcDEF"
        result = scrub_secrets(text)
        assert "xoxb-" not in result
        assert "[REDACTED]" in result

    def test_google_oauth_token(self) -> None:
        text = "token ya29.a0ARrdaM_abcdefghij1234"
        result = scrub_secrets(text)
        assert "ya29." not in result
        assert "[REDACTED]" in result

    def test_google_api_key(self) -> None:
        text = "key AIzaSy" + "a" * 33
        result = scrub_secrets(text)
        assert "AIzaSy" not in result
        assert "[REDACTED]" in result

    def test_empty_string(self) -> None:
        assert scrub_secrets("") == ""


class TestValidateSourcePath:
    def test_valid_project_path(self, tmp_path: Path) -> None:
        rule_file = tmp_path / "rules.txt"
        rule_file.touch()
        result = validate_source_path(
            "rules.txt", tmp_path, (), is_global=False
        )
        assert result == rule_file.resolve()

    def test_dotdot_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="contains '..'"):
            validate_source_path(
                "../evil.txt", tmp_path, (), is_global=False
            )

    def test_dotdot_substring_allowed(self, tmp_path: Path) -> None:
        """A filename containing '..' as substring (not path component) is allowed."""
        rule_file = tmp_path / "my..file.txt"
        rule_file.touch()
        result = validate_source_path(
            str(rule_file), tmp_path, (), is_global=False
        )
        assert result == rule_file.resolve()

    def test_absolute_path_in_project(self, tmp_path: Path) -> None:
        rule_file = tmp_path / "rules.txt"
        rule_file.touch()
        result = validate_source_path(
            str(rule_file), tmp_path, (), is_global=False
        )
        assert result == rule_file.resolve()

    def test_path_outside_project_rejected(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        project.mkdir()
        outside = tmp_path / "outside" / "evil.txt"
        outside.parent.mkdir()
        outside.touch()
        with pytest.raises(ConfigError, match="outside"):
            validate_source_path(
                str(outside), project, (), is_global=False
            )

    def test_allowed_dirs_project(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        project.mkdir()
        external = tmp_path / "external"
        external.mkdir()
        rule_file = external / "rules.txt"
        rule_file.touch()
        result = validate_source_path(
            str(rule_file),
            project,
            (str(external),),
            is_global=False,
        )
        assert result == rule_file.resolve()

    def test_global_path_under_cuecard(self, tmp_path: Path) -> None:
        """Global config: path under ~/.cuecard/ is allowed."""
        fake_home = tmp_path / "home"
        cuecard_dir = fake_home / ".cuecard"
        cuecard_dir.mkdir(parents=True)
        rules = cuecard_dir / "rules.txt"
        rules.touch()

        # Monkeypatch Path.home()
        original_home = Path.home

        def mock_home() -> Path:
            return fake_home

        Path.home = staticmethod(mock_home)  # type: ignore[assignment]
        try:
            result = validate_source_path(
                str(rules), cuecard_dir, (), is_global=True
            )
            assert result == rules.resolve()
        finally:
            Path.home = original_home  # type: ignore[assignment]

    def test_global_path_outside_cuecard_rejected(self, tmp_path: Path) -> None:
        fake_home = tmp_path / "home"
        cuecard_dir = fake_home / ".cuecard"
        cuecard_dir.mkdir(parents=True)
        outside = tmp_path / "other" / "rules.txt"
        outside.parent.mkdir()
        outside.touch()

        original_home = Path.home

        def mock_home() -> Path:
            return fake_home

        Path.home = staticmethod(mock_home)  # type: ignore[assignment]
        try:
            with pytest.raises(ConfigError, match="outside"):
                validate_source_path(
                    str(outside), cuecard_dir, (), is_global=True
                )
        finally:
            Path.home = original_home  # type: ignore[assignment]

    def test_global_allowed_dirs(self, tmp_path: Path) -> None:
        fake_home = tmp_path / "home"
        cuecard_dir = fake_home / ".cuecard"
        cuecard_dir.mkdir(parents=True)
        external = tmp_path / "external"
        external.mkdir()
        rules = external / "rules.txt"
        rules.touch()

        original_home = Path.home

        def mock_home() -> Path:
            return fake_home

        Path.home = staticmethod(mock_home)  # type: ignore[assignment]
        try:
            result = validate_source_path(
                str(rules),
                cuecard_dir,
                (str(external),),
                is_global=True,
            )
            assert result == rules.resolve()
        finally:
            Path.home = original_home  # type: ignore[assignment]

    def test_tilde_expansion(self, tmp_path: Path) -> None:
        fake_home = tmp_path / "home"
        cuecard_dir = fake_home / ".cuecard"
        cuecard_dir.mkdir(parents=True)
        rules = cuecard_dir / "rules.txt"
        rules.touch()

        original_expanduser = os.path.expanduser

        def mock_expanduser(p: str) -> str:
            return p.replace("~", str(fake_home))

        original_home = Path.home

        def mock_home() -> Path:
            return fake_home

        Path.home = staticmethod(mock_home)  # type: ignore[assignment]
        os.path.expanduser = mock_expanduser  # type: ignore[assignment]
        try:
            result = validate_source_path(
                "~/.cuecard/rules.txt",
                cuecard_dir,
                (),
                is_global=True,
            )
            assert result == rules.resolve()
        finally:
            os.path.expanduser = original_expanduser  # type: ignore[assignment]
            Path.home = original_home  # type: ignore[assignment]


class TestEnsureDirectory:
    def test_creates_dir(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "new_dir"
        ensure_directory(new_dir)
        assert new_dir.is_dir()
        mode = stat.S_IMODE(new_dir.stat().st_mode)
        assert mode == 0o700

    def test_existing_dir_ok(self, tmp_path: Path) -> None:
        existing = tmp_path / "existing"
        existing.mkdir()
        ensure_directory(existing)
        assert existing.is_dir()

    def test_nested_creation(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        ensure_directory(nested)
        assert nested.is_dir()

    def test_file_exists_error(self, tmp_path: Path) -> None:
        file_path = tmp_path / "not_a_dir"
        file_path.touch()
        with pytest.raises(ConfigError, match="Expected directory"):
            ensure_directory(file_path)


class TestSecureOpen:
    def test_create_new_file(self, tmp_path: Path) -> None:
        path = tmp_path / "new.txt"
        with secure_open(path, "w") as f:
            f.write("hello")
        assert path.read_text() == "hello"
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == 0o600

    def test_open_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "existing.txt"
        path.write_text("content")
        with secure_open(path, "r") as f:
            assert f.read() == "content"

    def test_warns_permissive_file(self, tmp_path: Path) -> None:
        path = tmp_path / "permissive.txt"
        path.write_text("data")
        path.chmod(0o644)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with secure_open(path, "r") as f:
                f.read()
            assert len(w) == 1
            assert "permissive" in str(w[0].message).lower()

    def test_read_nonexistent_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "missing.txt"
        with pytest.raises(FileNotFoundError):
            secure_open(path, "r")

    def test_append_new_file(self, tmp_path: Path) -> None:
        path = tmp_path / "append.txt"
        with secure_open(path, "a") as f:
            f.write("line1\n")
        assert path.read_text() == "line1\n"
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == 0o600

    def test_no_warn_secure_file(self, tmp_path: Path) -> None:
        path = tmp_path / "secure.txt"
        path.write_text("data")
        path.chmod(0o600)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with secure_open(path, "r") as f:
                f.read()
            assert len(w) == 0
