"""Thin HTTP client for the PermRAG API.

Every method here maps to exactly one endpoint in src/permrag/api/routers/.
Keeping the HTTP details in one place means the Streamlit layer only ever
deals with plain dicts and a single error type.
"""

from __future__ import annotations

from typing import Any

import httpx


class APIError(Exception):
    """A non-2xx response, carrying the server's own detail message."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"{status_code}: {detail}")


class PermRAGClient:
    def __init__(self, base_url: str, token: str | None = None, timeout: float = 90.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        # Chat waits on an embedding call plus a completion, so the default
        # httpx timeout of 5s is far too short.
        self._timeout = timeout

    # --- plumbing ------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self._base_url}{path}"
        try:
            response = httpx.request(
                method,
                url,
                json=json,
                params=params,
                headers=self._headers(),
                timeout=self._timeout,
            )
        except httpx.RequestError as exc:
            raise APIError(0, f"Could not reach the API at {url} ({exc})") from exc

        if response.status_code >= 400:
            raise APIError(response.status_code, _extract_detail(response))

        if response.status_code == 204 or not response.content:
            return None
        return response.json()

    # --- system --------------------------------------------------------------

    def health(self) -> dict:
        return self._request("GET", "/health")

    # --- auth ----------------------------------------------------------------

    def login(self, email: str, password: str) -> dict:
        return self._request("POST", "/auth/login", json={"email": email, "password": password})

    def me(self) -> dict:
        return self._request("GET", "/auth/me")

    # --- departments (admin) -------------------------------------------------

    def list_departments(self) -> list[dict]:
        return self._request("GET", "/admin/departments")

    def create_department(self, slug: str, name: str, description: str | None) -> dict:
        return self._request(
            "POST",
            "/admin/departments",
            json={"slug": slug, "name": name, "description": description},
        )

    def list_department_members(self, department_id: str) -> list[dict]:
        return self._request("GET", f"/admin/departments/{department_id}/members")

    # --- users (admin) -------------------------------------------------------

    def list_users(self) -> list[dict]:
        return self._request("GET", "/admin/users")

    def create_user(self, email: str, full_name: str, password: str, is_admin: bool) -> dict:
        return self._request(
            "POST",
            "/admin/users",
            json={
                "email": email,
                "full_name": full_name,
                "password": password,
                "is_admin": is_admin,
            },
        )

    def deactivate_user(self, user_id: str) -> dict:
        return self._request("POST", f"/admin/users/{user_id}/deactivate")

    # --- membership (admin) --------------------------------------------------

    def add_membership(self, user_id: str, department_id: str, is_manager: bool) -> dict:
        return self._request(
            "POST",
            "/admin/memberships",
            json={
                "user_id": user_id,
                "department_id": department_id,
                "is_manager": is_manager,
            },
        )

    def remove_membership(self, user_id: str, department_id: str) -> dict:
        # This endpoint takes query parameters, not a body.
        return self._request(
            "DELETE",
            "/admin/memberships",
            params={"user_id": user_id, "department_id": department_id},
        )

    # --- documents -----------------------------------------------------------

    def ingest_document(
        self,
        external_id: str,
        title: str,
        owner_department_id: str,
        pages: list[dict],
        source_uri: str | None = None,
    ) -> dict:
        return self._request(
            "POST",
            "/documents",
            json={
                "external_id": external_id,
                "title": title,
                "owner_department_id": owner_department_id,
                "source_uri": source_uri,
                "pages": pages,
            },
        )

    def list_documents(self) -> list[dict]:
        """Documents the *current* user may view, per SpiceDB."""
        return self._request("GET", "/documents")

    def get_document(self, document_id: str) -> dict:
        return self._request("GET", f"/documents/{document_id}")

    def delete_document(self, document_id: str) -> dict:
        return self._request("DELETE", f"/documents/{document_id}")

    # --- sharing (admin) -----------------------------------------------------

    def share_with_department(
        self, document_id: str, department_id: str, reason: str | None
    ) -> dict:
        return self._request(
            "POST",
            f"/documents/{document_id}/shares/departments",
            json={"department_id": department_id, "reason": reason},
        )

    def unshare_with_department(self, document_id: str, department_id: str) -> dict:
        return self._request(
            "DELETE", f"/documents/{document_id}/shares/departments/{department_id}"
        )

    def share_with_user(self, document_id: str, user_id: str, reason: str | None) -> dict:
        return self._request(
            "POST",
            f"/documents/{document_id}/shares/users",
            json={"user_id": user_id, "reason": reason},
        )

    def revoke_from_user(self, document_id: str, user_id: str) -> dict:
        return self._request("DELETE", f"/documents/{document_id}/shares/users/{user_id}")

    def list_document_access(self, document_id: str) -> dict:
        return self._request("GET", f"/documents/{document_id}/access")

    # --- chat ----------------------------------------------------------------

    def ask(self, question: str) -> dict:
        return self._request("POST", "/chat", json={"question": question})


def _extract_detail(response: httpx.Response) -> str:
    """Pull a readable message out of an error response.

    The app's own handlers always return {"detail": "..."}, but FastAPI's
    request validation returns a list of error objects under the same key.
    """
    try:
        payload = response.json()
    except ValueError:
        return response.text or f"HTTP {response.status_code}"

    detail = payload.get("detail", response.text)

    if isinstance(detail, list):
        parts = []
        for item in detail:
            location = ".".join(str(p) for p in item.get("loc", []) if p != "body")
            parts.append(f"{location}: {item.get('msg', '')}".strip(": "))
        return "; ".join(parts)

    return str(detail)
