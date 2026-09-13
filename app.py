import os
import uuid
import time
import json
import sqlite3
import hashlib
import secrets
from datetime import datetime
import streamlit as st
from groq import Groq, APIStatusError, APIConnectionError

st.set_page_config(page_title="ChatSpark", page_icon=":material/bolt:", layout="wide")

api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
client = Groq(api_key=api_key)

MODEL_OPTIONS = {
    "GPT-OSS 20B (fast)": "openai/gpt-oss-20b",
    "GPT-OSS 120B (smarter)": "openai/gpt-oss-120b",
}

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DB_PATH = "chatspark.db"


# ---------- Persistence layer (SQLite) ----------

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            salt TEXT,
            password_hash TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            username TEXT,
            title TEXT,
            system_prompt TEXT,
            messages TEXT,
            updated_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            username TEXT,
            created_at TEXT
        )
    """)
    conn.commit()

    # Migrate older DBs created before the `username` column existed
    existing_cols = [row[1] for row in conn.execute("PRAGMA table_info(chats)").fetchall()]
    if "username" not in existing_cols:
        conn.execute("ALTER TABLE chats ADD COLUMN username TEXT")
        conn.commit()

    return conn


def create_session(conn, username):
    token = secrets.token_hex(24)
    conn.execute(
        "INSERT INTO sessions (token, username, created_at) VALUES (?, ?, ?)",
        (token, username, datetime.now().isoformat()),
    )
    conn.commit()
    return token


def get_session_user(conn, token):
    row = conn.execute(
        "SELECT username FROM sessions WHERE token = ?", (token,)
    ).fetchone()
    return row[0] if row else None


def delete_session(conn, token):
    conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
    conn.commit()


def hash_password(password, salt):
    return hashlib.sha256((salt + password).encode()).hexdigest()


def create_user(conn, username, password):
    salt = secrets.token_hex(16)
    pw_hash = hash_password(password, salt)
    try:
        conn.execute(
            "INSERT INTO users (username, salt, password_hash) VALUES (?, ?, ?)",
            (username, salt, pw_hash),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def verify_user(conn, username, password):
    row = conn.execute(
        "SELECT salt, password_hash FROM users WHERE username = ?", (username,)
    ).fetchone()
    if not row:
        return False
    salt, stored_hash = row
    return hash_password(password, salt) == stored_hash


def save_chat(conn, chat_id, username, chat):
    conn.execute(
        "REPLACE INTO chats (id, username, title, system_prompt, messages, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (
            chat_id,
            username,
            chat["title"],
            chat["messages"][0]["content"],
            json.dumps(chat["messages"]),
            datetime.now().isoformat(),
        ),
    )
    conn.commit()


def delete_chat_db(conn, chat_id):
    conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    conn.commit()


def load_user_chats(conn, username):
    rows = conn.execute(
        "SELECT id, title, messages FROM chats WHERE username = ? ORDER BY updated_at DESC",
        (username,),
    ).fetchall()
    return {row[0]: {"title": row[1], "messages": json.loads(row[2])} for row in rows}


conn = get_db()


# ---------- Auth (guest by default, login/signup optional via sidebar) ----------

if "user" not in st.session_state:
    session_token = st.query_params.get("session")
    restored_user = get_session_user(conn, session_token) if session_token else None

    if restored_user:
        st.session_state.user = restored_user
        st.session_state.is_guest = False
    else:
        st.session_state.user = f"guest_{uuid.uuid4().hex[:8]}"
        st.session_state.is_guest = True

username = st.session_state.user


# ---------- App state (scoped to logged-in user) ----------

if "chats" not in st.session_state:
    if st.session_state.get("is_guest"):
        st.session_state.chats = {}
    else:
        st.session_state.chats = load_user_chats(conn, username)
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
if "renaming" not in st.session_state:
    st.session_state.renaming = None
if "search_query" not in st.session_state:
    st.session_state.search_query = ""


def persist_chat(chat_id, chat):
    if not st.session_state.get("is_guest"):
        save_chat(conn, chat_id, username, chat)


def persist_delete(chat_id):
    if not st.session_state.get("is_guest"):
        delete_chat_db(conn, chat_id)


def new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "title": "New Chat",
        "messages": [{"role": "system", "content": st.session_state.system_prompt}],
    }
    st.session_state.active_chat = chat_id
    persist_chat(chat_id, st.session_state.chats[chat_id])


if "active_chat" not in st.session_state or st.session_state.active_chat not in st.session_state.chats:
    if st.session_state.chats:
        st.session_state.active_chat = next(iter(st.session_state.chats))
    else:
        new_chat()


def run_completion(messages, model, temperature):
    try:
        api_messages = [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]
        stream = client.chat.completions.create(
            model=model,
            messages=api_messages,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            yield chunk.choices[0].delta.content or ""
    except APIStatusError as e:
        if e.status_code == 429:
            raise RuntimeError(
                "Rate limit reached. Groq's free tier caps requests per minute — "
                "wait a bit and try again, or switch to a lighter model in the sidebar."
            )
        elif e.status_code == 401:
            raise RuntimeError("Invalid or missing API key. Check your GROQ_API_KEY.")
        elif e.status_code == 404:
            raise RuntimeError(f"Model '{model}' isn't available on your account.")
        else:
            raise RuntimeError(f"Groq API error ({e.status_code}): {e.message}")
    except APIConnectionError:
        raise RuntimeError("Couldn't reach Groq's servers. Check your connection and try again.")


# ---------- Sidebar ----------

with st.sidebar:
    if st.session_state.get("is_guest"):
        st.caption("Browsing as **guest** · chats won't be saved")
        with st.popover("Log in / Sign up", use_container_width=True, icon=":material/login:"):
            login_tab, signup_tab = st.tabs(["Log in", "Sign up"])

            with login_tab:
                with st.form("login_form"):
                    login_username = st.text_input("Username")
                    login_password = st.text_input("Password", type="password")
                    submitted = st.form_submit_button("Log in", use_container_width=True)
                    if submitted:
                        if verify_user(conn, login_username, login_password):
                            token = create_session(conn, login_username)
                            st.query_params["session"] = token
                            st.session_state.user = login_username
                            st.session_state.pop("is_guest", None)
                            st.session_state.pop("chats", None)
                            st.session_state.pop("active_chat", None)
                            st.rerun()
                        else:
                            st.error("Incorrect username or password.")

            with signup_tab:
                with st.form("signup_form"):
                    new_username = st.text_input("Choose a username")
                    new_password = st.text_input("Choose a password", type="password")
                    signup_submitted = st.form_submit_button("Create account", use_container_width=True)
                    if signup_submitted:
                        if not new_username or not new_password:
                            st.error("Username and password can't be empty.")
                        elif create_user(conn, new_username, new_password):
                            token = create_session(conn, new_username)
                            st.query_params["session"] = token
                            st.session_state.user = new_username
                            st.session_state.pop("is_guest", None)
                            st.session_state.pop("chats", None)
                            st.session_state.pop("active_chat", None)
                            st.rerun()
                        else:
                            st.error("That username is already taken.")
    else:
        st.caption(f"Logged in as **{username}**")
        if st.button("Log out", use_container_width=True, icon=":material/logout:"):
            token = st.query_params.get("session")
            if token:
                delete_session(conn, token)
                del st.query_params["session"]
            st.session_state.user = f"guest_{uuid.uuid4().hex[:8]}"
            st.session_state.is_guest = True
            st.session_state.pop("chats", None)
            st.session_state.pop("active_chat", None)
            st.rerun()

    st.divider()
    st.subheader("Chats")
    if st.button("New Chat", use_container_width=True, icon=":material/add:"):
        new_chat()
        st.rerun()

    st.session_state.search_query = st.text_input(
        "Search chats", value=st.session_state.search_query,
        placeholder="Search...", label_visibility="collapsed",
        icon=":material/search:",
    )

    query = st.session_state.search_query.lower().strip()

    def chat_matches(chat):
        if not query:
            return True
        if query in chat["title"].lower():
            return True
        return any(query in m["content"].lower() for m in chat["messages"] if m["role"] != "system")

    visible_chats = {
        cid: c for cid, c in st.session_state.chats.items() if chat_matches(c)
    }

    for cid, chat in visible_chats.items():
        if st.session_state.renaming == cid:
            new_title = st.text_input(
                "Rename", value=chat["title"], key=f"rename_input_{cid}", label_visibility="collapsed"
            )
            rcols = st.columns([1, 1])
            with rcols[0]:
                if st.button("Save", key=f"save_{cid}", use_container_width=True, icon=":material/check:"):
                    chat["title"] = new_title or chat["title"]
                    persist_chat(cid, chat)
                    st.session_state.renaming = None
                    st.rerun()
            with rcols[1]:
                if st.button("Cancel", key=f"cancel_{cid}", use_container_width=True, icon=":material/close:"):
                    st.session_state.renaming = None
                    st.rerun()
        else:
            cols = st.columns([3, 1, 1])
            with cols[0]:
                active_marker = ":material/radio_button_checked:" if cid == st.session_state.active_chat else ":material/chat_bubble:"
                if st.button(
                    chat["title"], key=f"select_{cid}", use_container_width=True,
                    icon=active_marker,
                ):
                    st.session_state.active_chat = cid
                    st.rerun()
            with cols[1]:
                if st.button("", key=f"rename_{cid}", icon=":material/edit:"):
                    st.session_state.renaming = cid
                    st.rerun()
            with cols[2]:
                if st.button("", key=f"delete_{cid}", icon=":material/delete:"):
                    persist_delete(cid)
                    del st.session_state.chats[cid]
                    if st.session_state.active_chat == cid:
                        if st.session_state.chats:
                            st.session_state.active_chat = next(iter(st.session_state.chats))
                        else:
                            new_chat()
                    st.rerun()

    if query and not visible_chats:
        st.caption("No chats match your search.")

    st.divider()
    st.subheader("Settings")
    selected_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()))
    MODEL = MODEL_OPTIONS[selected_label]

    temperature = st.slider("Creativity (temperature)", 0.0, 1.5, 0.7, 0.1)

    new_system_prompt = st.text_area(
        "System prompt", value=st.session_state.system_prompt, height=100
    )
    if new_system_prompt != st.session_state.system_prompt:
        st.session_state.system_prompt = new_system_prompt
        st.session_state.chats[st.session_state.active_chat]["messages"][0] = {
            "role": "system",
            "content": new_system_prompt,
        }
        persist_chat(st.session_state.active_chat, st.session_state.chats[st.session_state.active_chat])

    st.divider()
    active = st.session_state.chats[st.session_state.active_chat]
    chat_text = "\n\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in active["messages"]
        if m["role"] != "system"
    )
    st.download_button(
        "Export chat",
        data=chat_text,
        file_name=f"chat_{st.session_state.active_chat[:8]}.txt",
        mime="text/plain",
        use_container_width=True,
        icon=":material/download:",
    )

# ---------- Main chat area ----------

st.title("ChatSpark")
st.caption("Developed By Wasif Azhar")

active_chat = st.session_state.chats[st.session_state.active_chat]
messages = active_chat["messages"]

for msg in messages:
    if msg["role"] != "system":
        avatar = ":material/person:" if msg["role"] == "user" else ":material/bolt:"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if "timestamp" in msg:
                st.caption(msg["timestamp"])

has_assistant_reply = any(m["role"] == "assistant" for m in messages)

if has_assistant_reply:
    if st.button("Regenerate response", icon=":material/refresh:"):
        while messages and messages[-1]["role"] != "user":
            messages.pop()
        st.session_state.regenerate = True
        st.rerun()

prompt = st.chat_input("Type your message...")

if prompt:
    messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().strftime("%H:%M"),
    })
    if active_chat["title"] == "New Chat":
        active_chat["title"] = prompt[:30] + ("..." if len(prompt) > 30 else "")
    persist_chat(st.session_state.active_chat, active_chat)

if prompt or st.session_state.get("regenerate"):
    st.session_state.regenerate = False

    if prompt:
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(prompt)

    with st.chat_message("assistant", avatar=":material/bolt:"):
        placeholder = st.empty()
        partial = ""
        start_time = time.time()
        error_msg = None

        try:
            for delta in run_completion(messages, MODEL, temperature):
                partial += delta
                placeholder.markdown(partial + "▌")
            placeholder.markdown(partial)
        except RuntimeError as e:
            error_msg = str(e)
            placeholder.error(error_msg)

        if not error_msg:
            elapsed = time.time() - start_time
            word_count = len(partial.split())
            st.caption(f"{word_count} words · {elapsed:.1f}s")

    if not error_msg:
        messages.append({
            "role": "assistant",
            "content": partial,
            "timestamp": datetime.now().strftime("%H:%M"),
        })
        persist_chat(st.session_state.active_chat, active_chat)
    else:
        if messages and messages[-1]["role"] == "user":
            messages.pop()
