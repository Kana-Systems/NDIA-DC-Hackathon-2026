"""Fail-closed runtime secrets replaced with deterministic test-only values."""

from __future__ import annotations

import os

os.environ.setdefault("WORKSPACE_PASSWORD", "pytest-only-password")
os.environ.setdefault("DEMO_JWT_SECRET", "pytest-only-signing-secret-32-characters")
