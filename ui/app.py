"""Streamlit console for PermRAG.

Run with:
    uv run streamlit run ui/app.py

This talks to the FastAPI service over HTTP exactly the way any other client
would -- it holds no database, SpiceDB, or Qdrant connection of its own. That
matters: every permission decision shown here was made by the API, not by the
UI, so what you see is genuinely what the permission system returned.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from api_client import APIError, PermRAGClient  # noqa: E402

DEFAULT_API_URL = os.getenv("PERMRAG_API_URL", "http://localhost:8000")

st.set_page_config(page_title="PermRAG Console", page_icon="🔐", layout="wide")


# --- session helpers ---------------------------------------------------------


def _init_state() -> None:
    st.session_state.setdefault("token", None)
    st.session_state.setdefault("user", None)
    st.session_state.setdefault("api_url", DEFAULT_API_URL)
    st.session_state.setdefault("chat_history", [])


def client() -> PermRAGClient:
    return PermRAGClient(st.session_state["api_url"], st.session_state.get("token"))


def logout() -> None:
    st.session_state["token"] = None
    st.session_state["user"] = None
    st.session_state["chat_history"] = []


def show_error(exc: APIError) -> None:
    """Render an API failure with the distinctions that actually matter here."""
    if exc.status_code == 401:
        st.error("Session expired or not authorised. Please sign in again.")
        logout()
    elif exc.status_code == 403:
        st.error("Your account is not an admin, so this action is not available to you.")
    elif exc.status_code == 404:
        # The API deliberately returns 404 rather than 403 for documents you
        # cannot see, so this is not necessarily "missing".
        st.warning(f"Not found, or not visible to your account. ({exc.detail})")
    elif exc.status_code == 503:
        st.error(
            "The permission system is unreachable, so the request was refused. "
            "PermRAG fails closed by design rather than returning unfiltered results."
        )
    elif exc.status_code == 0:
        st.error(exc.detail)
    else:
        st.error(f"{exc.status_code}: {exc.detail}")


def label_for_user(user: dict) -> str:
    suffix = " [admin]" if user.get("is_admin") else ""
    inactive = "" if user.get("is_active", True) else " (inactive)"
    return f"{user['full_name']} <{user['email']}>{suffix}{inactive}"


def label_for_department(dept: dict) -> str:
    return f"{dept['name']} ({dept['slug']})"


def label_for_document(doc: dict) -> str:
    return f"{doc['title']} [{doc['external_id']}]"


@st.cache_data(ttl=10, show_spinner=False)
def _cached_lookup(token: str, api_url: str, kind: str) -> list[dict]:
    """Short-lived cache for dropdown data.

    Keyed on the token so one user's lists never leak into another's session
    after a logout/login.
    """
    api = PermRAGClient(api_url, token)
    if kind == "departments":
        return api.list_departments()
    if kind == "users":
        return api.list_users()
    if kind == "documents":
        return api.list_documents()
    raise ValueError(kind)


def lookup(kind: str) -> list[dict]:
    return _cached_lookup(st.session_state["token"], st.session_state["api_url"], kind)


def clear_lookups() -> None:
    _cached_lookup.clear()


# --- login -------------------------------------------------------------------


def render_login() -> None:
    st.title("🔐 PermRAG Console")
    st.caption("Permission-aware retrieval. Every answer is filtered by SpiceDB.")

    with st.form("login"):
        api_url = st.text_input("API URL", value=st.session_state["api_url"])
        email = st.text_input("Email", value="admin@example.com")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", use_container_width=True)

    if submitted:
        st.session_state["api_url"] = api_url.strip()
        try:
            api = PermRAGClient(st.session_state["api_url"])
            tokens = api.login(email.strip(), password)
            st.session_state["token"] = tokens["access_token"]
            st.session_state["user"] = PermRAGClient(
                st.session_state["api_url"], tokens["access_token"]
            ).me()
            clear_lookups()
            st.rerun()
        except APIError as exc:
            if exc.status_code == 401:
                st.error("Incorrect email or password.")
            else:
                show_error(exc)

    with st.expander("Can't connect?"):
        st.markdown(
            "- Make sure the API is running: `uv run uvicorn permrag.api.main:app --reload`\n"
            "- Make sure SpiceDB and Qdrant are up: `docker compose ps`\n"
            "- The admin account is the one created by `scripts/bootstrap.py`"
        )


# --- chat --------------------------------------------------------------------


def render_chat() -> None:
    st.header("Chat")
    st.caption(
        "Answers are drawn only from documents your account is permitted to see. "
        "Sign in as different users to watch the same question return different answers."
    )

    for entry in st.session_state["chat_history"]:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.write(entry["answer"])
            _render_answer_meta(entry)

    question = st.chat_input("Ask a question...")
    if not question:
        return

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Checking permissions, retrieving, answering..."):
            try:
                result = client().ask(question)
            except APIError as exc:
                show_error(exc)
                return

        st.write(result["answer"])
        _render_answer_meta(result)

    st.session_state["chat_history"].append({"question": question, **result})


def _render_answer_meta(result: dict) -> None:
    """Show the permission-relevant numbers alongside every answer.

    These three counts are the whole point of the system: how many documents
    the permission graph cleared, and how many chunks survived filtering.
    """
    cols = st.columns(3)
    cols[0].metric("Permitted documents", result.get("permitted_document_count", 0))
    cols[1].metric("Chunks retrieved", result.get("retrieved_chunk_count", 0))
    cols[2].metric("Latency", f"{result.get('latency_ms', 0)} ms")

    citations = result.get("citations") or []
    if citations:
        with st.expander(f"Citations ({len(citations)})"):
            for citation in citations:
                line = f"**[{citation['index']}]** {citation['title']} — page {citation['page_number']}"
                if citation.get("source_uri"):
                    line += f" — {citation['source_uri']}"
                st.markdown(line)
                st.caption(f"document_id: `{citation['document_id']}`")

    if result.get("trace_id"):
        st.caption(f"Langfuse trace: `{result['trace_id']}`")


# --- my documents ------------------------------------------------------------


def render_my_documents() -> None:
    st.header("My documents")
    st.caption(
        "This list comes from SpiceDB first -- Postgres is only used to fill in "
        "titles for ids the permission system already cleared."
    )

    if st.button("Refresh"):
        clear_lookups()

    try:
        documents = lookup("documents")
    except APIError as exc:
        show_error(exc)
        return

    if not documents:
        st.info("You do not currently have access to any documents.")
        return

    st.success(f"You can see {len(documents)} document(s).")
    st.dataframe(
        [
            {
                "Title": d["title"],
                "External ID": d["external_id"],
                "Status": d["status"],
                "Indexed": d.get("indexed_at") or "—",
                "Document ID": d["id"],
            }
            for d in documents
        ],
        use_container_width=True,
        hide_index=True,
    )


# --- departments (admin) -----------------------------------------------------


def render_departments() -> None:
    st.header("Departments")

    with st.expander("Create a department"):
        with st.form("create_department"):
            slug = st.text_input("Slug", placeholder="engineering")
            st.caption("Lowercase letters, numbers and hyphens; must start alphanumeric.")
            name = st.text_input("Name", placeholder="Engineering")
            description = st.text_area("Description (optional)")
            if st.form_submit_button("Create"):
                try:
                    dept = client().create_department(
                        slug.strip(), name.strip(), description.strip() or None
                    )
                    clear_lookups()
                    st.success(f"Created '{dept['name']}'")
                    st.rerun()
                except APIError as exc:
                    show_error(exc)

    try:
        departments = lookup("departments")
    except APIError as exc:
        show_error(exc)
        return

    if not departments:
        st.info("No departments yet.")
        return

    st.dataframe(
        [
            {
                "Name": d["name"],
                "Slug": d["slug"],
                "Description": d.get("description") or "—",
                "ID": d["id"],
            }
            for d in departments
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Members")
    chosen = st.selectbox(
        "Department", departments, format_func=label_for_department, key="dept_members"
    )
    if chosen:
        try:
            members = client().list_department_members(chosen["id"])
        except APIError as exc:
            show_error(exc)
            return
        if members:
            st.dataframe(
                [
                    {
                        "Name": m["full_name"],
                        "Email": m["email"],
                        "Admin": m["is_admin"],
                        "Active": m["is_active"],
                    }
                    for m in members
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No members in this department.")


# --- users (admin) -----------------------------------------------------------


def render_users() -> None:
    st.header("Users")

    with st.expander("Create a user"):
        with st.form("create_user"):
            email = st.text_input("Email")
            full_name = st.text_input("Full name")
            password = st.text_input("Password", type="password")
            st.caption("Between 8 and 72 characters.")
            is_admin = st.checkbox("Grant admin rights")
            if st.form_submit_button("Create"):
                try:
                    user = client().create_user(
                        email.strip(), full_name.strip(), password, is_admin
                    )
                    clear_lookups()
                    st.success(f"Created {user['email']}")
                    st.rerun()
                except APIError as exc:
                    show_error(exc)

    try:
        users = lookup("users")
        departments = lookup("departments")
    except APIError as exc:
        show_error(exc)
        return

    if users:
        st.dataframe(
            [
                {
                    "Name": u["full_name"],
                    "Email": u["email"],
                    "Admin": u["is_admin"],
                    "Active": u["is_active"],
                    "ID": u["id"],
                }
                for u in users
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Department membership")
    st.caption(
        "Granting membership writes a relationship to SpiceDB and returns a ZedToken, "
        "which is what makes the change visible to the very next query."
    )

    col_add, col_remove = st.columns(2)

    with col_add:
        st.markdown("**Grant**")
        user_add = st.selectbox("User", users, format_func=label_for_user, key="mem_add_user")
        dept_add = st.selectbox(
            "Department", departments, format_func=label_for_department, key="mem_add_dept"
        )
        is_manager = st.checkbox("As manager", key="mem_add_mgr")
        if st.button("Grant membership", use_container_width=True):
            if user_add and dept_add:
                try:
                    result = client().add_membership(
                        user_add["id"], dept_add["id"], is_manager
                    )
                    clear_lookups()
                    st.success("Membership granted.")
                    st.caption(f"ZedToken: `{result['zed_token']}`")
                except APIError as exc:
                    show_error(exc)

    with col_remove:
        st.markdown("**Revoke**")
        user_rm = st.selectbox("User", users, format_func=label_for_user, key="mem_rm_user")
        dept_rm = st.selectbox(
            "Department", departments, format_func=label_for_department, key="mem_rm_dept"
        )
        if st.button("Revoke membership", use_container_width=True):
            if user_rm and dept_rm:
                try:
                    client().remove_membership(user_rm["id"], dept_rm["id"])
                    clear_lookups()
                    st.success("Membership revoked.")
                except APIError as exc:
                    show_error(exc)

    st.subheader("Deactivate an account")
    st.caption(
        "Deactivation also purges every SpiceDB relationship the account holds, "
        "so access does not come back if the account is later re-enabled."
    )
    user_off = st.selectbox("User", users, format_func=label_for_user, key="deact_user")
    confirm = st.checkbox("I understand this revokes all of their access", key="deact_confirm")
    if st.button("Deactivate", type="primary", disabled=not confirm):
        if user_off:
            try:
                result = client().deactivate_user(user_off["id"])
                clear_lookups()
                st.success(result["message"])
            except APIError as exc:
                show_error(exc)


# --- ingest (admin) ----------------------------------------------------------


def render_ingest() -> None:
    st.header("Ingest a document")
    st.caption(
        "Re-posting the same external ID performs an incremental update: pages whose "
        "content hash is unchanged are skipped without spending an embedding call."
    )

    try:
        departments = lookup("departments")
    except APIError as exc:
        show_error(exc)
        return

    if not departments:
        st.warning("Create a department first -- every document needs an owning department.")
        return

    external_id = st.text_input("External ID", placeholder="handbook-001")
    title = st.text_input("Title", placeholder="Engineering Handbook")
    department = st.selectbox(
        "Owning department", departments, format_func=label_for_department, key="ingest_dept"
    )
    source_uri = st.text_input("Source URI (optional)")

    st.markdown("**Pages**")
    mode = st.radio(
        "Input mode",
        ["Paste text", "Upload .txt files"],
        horizontal=True,
        label_visibility="collapsed",
    )

    pages: list[dict] = []

    if mode == "Paste text":
        page_count = st.number_input("Number of pages", min_value=1, max_value=50, value=1)
        for index in range(int(page_count)):
            content = st.text_area(f"Page {index + 1}", key=f"page_{index}", height=140)
            if content.strip():
                pages.append({"page_number": index + 1, "content": content})
    else:
        uploads = st.file_uploader(
            "One file per page, ordered by filename",
            type=["txt", "md"],
            accept_multiple_files=True,
        )
        for index, upload in enumerate(uploads or []):
            text = upload.read().decode("utf-8", errors="replace")
            if text.strip():
                pages.append({"page_number": index + 1, "content": text})
        if pages:
            st.caption(f"{len(pages)} page(s) ready.")

    if st.button("Ingest", type="primary", disabled=not pages):
        if not external_id.strip() or not title.strip():
            st.warning("External ID and title are both required.")
            return
        with st.spinner("Chunking, embedding, and writing permissions..."):
            try:
                result = client().ingest_document(
                    external_id=external_id.strip(),
                    title=title.strip(),
                    owner_department_id=department["id"],
                    pages=pages,
                    source_uri=source_uri.strip() or None,
                )
            except APIError as exc:
                show_error(exc)
                return

        clear_lookups()
        st.success("Document ingested.")
        cols = st.columns(5)
        cols[0].metric("Pages total", result["pages_total"])
        cols[1].metric("Indexed", result["pages_indexed"])
        cols[2].metric("Skipped", result["pages_skipped"])
        cols[3].metric("Removed", result["pages_removed"])
        cols[4].metric("Chunks written", result["chunks_written"])
        st.caption(f"document_id: `{result['document_id']}`")


# --- sharing and access (admin) ----------------------------------------------


def render_sharing() -> None:
    st.header("Sharing & access")

    try:
        documents = lookup("documents")
        departments = lookup("departments")
        users = lookup("users")
    except APIError as exc:
        show_error(exc)
        return

    if not documents:
        st.info("No documents visible to your account yet.")
        return

    document = st.selectbox(
        "Document", documents, format_func=label_for_document, key="share_doc"
    )
    if not document:
        return

    st.divider()
    st.subheader("Who has access right now")
    st.caption("Read straight from the permission graph, not from Postgres.")

    if st.button("Load access list"):
        try:
            access = client().list_document_access(document["id"])
        except APIError as exc:
            show_error(exc)
        else:
            entries = access.get("entries") or []
            if entries:
                st.dataframe(
                    [
                        {
                            "Relation": e["relation"],
                            "Subject type": e["subject_type"],
                            "Subject ID": e["subject_id"],
                        }
                        for e in entries
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No relationships recorded for this document.")

    st.divider()
    col_dept, col_user = st.columns(2)

    with col_dept:
        st.subheader("Cross-department share")
        dept = st.selectbox(
            "Department", departments, format_func=label_for_department, key="share_dept"
        )
        reason = st.text_input("Reason (optional)", key="share_dept_reason")
        c1, c2 = st.columns(2)
        if c1.button("Grant", key="grant_dept", use_container_width=True):
            if dept:
                try:
                    result = client().share_with_department(
                        document["id"], dept["id"], reason.strip() or None
                    )
                    st.success(result["message"])
                    st.caption(f"ZedToken: `{result['zed_token']}`")
                except APIError as exc:
                    show_error(exc)
        if c2.button("Revoke", key="revoke_dept", use_container_width=True):
            if dept:
                try:
                    result = client().unshare_with_department(document["id"], dept["id"])
                    st.success(result["message"])
                except APIError as exc:
                    show_error(exc)

    with col_user:
        st.subheader("Person-level override")
        user = st.selectbox("User", users, format_func=label_for_user, key="share_user")
        reason_u = st.text_input("Reason (optional)", key="share_user_reason")
        c3, c4 = st.columns(2)
        if c3.button("Grant", key="grant_user", use_container_width=True):
            if user:
                try:
                    result = client().share_with_user(
                        document["id"], user["id"], reason_u.strip() or None
                    )
                    st.success(result["message"])
                    st.caption(f"ZedToken: `{result['zed_token']}`")
                except APIError as exc:
                    show_error(exc)
        if c4.button("Revoke", key="revoke_user", use_container_width=True):
            if user:
                try:
                    result = client().revoke_from_user(document["id"], user["id"])
                    st.success(result["message"])
                except APIError as exc:
                    show_error(exc)

    st.divider()
    st.subheader("Delete document")
    st.caption("Permissions are removed before content, so nothing is briefly orphaned.")
    confirm = st.checkbox("I understand this cannot be undone", key="del_confirm")
    if st.button("Delete document", type="primary", disabled=not confirm):
        try:
            result = client().delete_document(document["id"])
            clear_lookups()
            st.success(result["message"])
            st.rerun()
        except APIError as exc:
            show_error(exc)


# --- shell -------------------------------------------------------------------


def render_sidebar() -> str:
    user = st.session_state["user"]

    with st.sidebar:
        st.markdown(f"### {user['full_name']}")
        st.caption(user["email"])
        if user["is_admin"]:
            st.caption("Administrator")

        st.divider()

        pages = ["Chat", "My documents"]
        if user["is_admin"]:
            pages += ["Departments", "Users", "Ingest", "Sharing & access"]

        choice = st.radio("Navigation", pages, label_visibility="collapsed")

        st.divider()
        try:
            health = client().health()
            st.caption(f"API {health['version']} · {health['environment']}")
        except APIError:
            st.caption("API unreachable")

        if st.button("Sign out", use_container_width=True):
            logout()
            clear_lookups()
            st.rerun()

    return choice


def main() -> None:
    _init_state()

    if not st.session_state.get("token"):
        render_login()
        return

    choice = render_sidebar()

    if choice == "Chat":
        render_chat()
    elif choice == "My documents":
        render_my_documents()
    elif choice == "Departments":
        render_departments()
    elif choice == "Users":
        render_users()
    elif choice == "Ingest":
        render_ingest()
    elif choice == "Sharing & access":
        render_sharing()


main()
