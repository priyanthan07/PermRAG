"""Test configuration.

Sets the environment variables Settings requires before anything imports it,
so unit tests run without a .env file present.

These are plain assignments, not ``setdefault``, on purpose. DeepEval ships a
pytest plugin that is imported before this file, and importing deepeval copies
every key from the project's .env into os.environ. With ``setdefault`` the real
.env values would win, and the suite would send traces to the live Langfuse
project and load real secrets.
"""

import os

os.environ["OPENAI_API_KEY"] = "sk-test-not-used"
os.environ["GEMINI_API_KEY"] = "gemini-test-not-used"
os.environ["QDRANT_API_KEY"] = ""
os.environ["JWT_SECRET_KEY"] = "test-secret-key-not-used-in-production"
os.environ["POSTGRES_PASSWORD"] = "test"
os.environ["LANGFUSE_ENABLED"] = "false"
os.environ["EVAL_ENABLED"] = "false"
# Unit tests must not run the cross-encoder model.
os.environ["RERANKER_ENABLED"] = "false"
