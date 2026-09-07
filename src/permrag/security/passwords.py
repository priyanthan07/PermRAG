import bcrypt

# bcrypt truncates silently past 72 bytes; reject rather than accept a
# password whose tail is ignored.
MAX_PASSWORD_BYTES = 72

def hash_password(plain_password: str) -> str:
    encoded = plain_password.encode("utf-8")
    if len(encoded) > MAX_PASSWORD_BYTES:
        raise ValueError(f"Password exceeds {MAX_PASSWORD_BYTES} bytes once UTF-8 encoded")
    return bcrypt.hashpw(encoded, bcrypt.gensalt()).decode("utf-8")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    encoded = plain_password.encode("utf-8")
    if len(encoded) > MAX_PASSWORD_BYTES:
        return False
    try:
        return bcrypt.checkpw(encoded, hashed_password.encode("utf-8"))
    except (ValueError, TypeError):
        # Malformed stored hash. Treat as a failed login, not a crash.
        return False
    