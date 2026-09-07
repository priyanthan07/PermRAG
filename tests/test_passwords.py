"""Password hashing. Uses bcrypt directly; passlib is broken on bcrypt 5.x."""

import pytest

from permrag.security.passwords import MAX_PASSWORD_BYTES, hash_password, verify_password


def test_round_trip():
    hashed = hash_password("correct-horse-battery")
    assert verify_password("correct-horse-battery", hashed)


def test_wrong_password_rejected():
    hashed = hash_password("correct-horse-battery")
    assert not verify_password("wrong-password", hashed)


def test_hash_is_salted():
    assert hash_password("same") != hash_password("same")


def test_malformed_stored_hash_returns_false_not_raises():
    assert not verify_password("anything", "not-a-real-bcrypt-hash")


def test_overlong_password_rejected_rather_than_truncated():
    """bcrypt silently ignores bytes past 72; accepting one would mean the
    tail of the password never protects anything."""
    with pytest.raises(ValueError):
        hash_password("x" * (MAX_PASSWORD_BYTES + 1))
        