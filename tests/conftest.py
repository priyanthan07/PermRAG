"""Test configuration.

Sets the environment variables Settings requires before anything imports it,
so unit tests run without a .env file present.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-used")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-not-used-in-production")
os.environ.setdefault("LANGFUSE_ENABLED", "false")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
