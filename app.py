import os
import uuid
import time
import streamlit as st
from groq import Groq

st.set_page_config(page_title="ChatSpark", page_icon="🤖", layout="wide")

api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
client = Groq(api_key=api_key)

MODEL_OPTIONS = {
    "GPT-OSS 20B (fast)": "openai/gpt-oss-20b",
    "GPT-OSS 120B (smarter)": "openai/gpt-oss-120b",
}

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "title": "New Chat",
        "messages": [{"role": "system", "content": st.session_state.system_prompt}],
    }
    st.session_state.active_chat = chat_id


if "chats" not in st.session_state:
    st.session_state.chats = {}
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
if "active_chat" not in st.session_state or st.session_state.active_chat not in st.session_state.chats:
    new_chat()

with st.sidebar:
    st.subheader("💬 Chats")
    if st.button("➕ New Chat", use_container_width=True):
        new_chat()
        st.rerun()

    for cid, chat in sorted(
        st.session_state.chats.items(), key=lambda x: x[0], reverse=True
    ):
        cols = st.columns([4, 1])
        with cols[0]:
            if st.button(chat["title"], key=f"select_{cid}", use_container_width=True):
                st.session_state.active_chat = cid
                st.rerun()
        with cols[1]:
            if st.button("🗑️", key=f"delete_{cid}"):
                del st.session_state.chats[cid]
                if st.session_state.active_chat == cid:
                    if st.session_state.chats:
                        st.session_state.active_chat = next(iter(st.session_state.chats))
                    else:
                        new_chat()
                st.rerun()

    st.divider()
    st.subheader("⚙️ Settings")
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

    st.divider()
    active = st.session_state.chats[st.session_state.active_chat]
    chat_text = "\n\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in active["messages"]
        if m["role"] != "system"
    )
    st.download_button(
        "⬇️ Export chat",
        data=chat_text,
        file_name=f"chat_{st.session_state.active_chat[:8]}.txt",
        mime="text/plain",
        use_container_width=True,
    )

st.title("🤖 AI Chatbot")

active_chat = st.session_state.chats[st.session_state.active_chat]
messages = active_chat["messages"]

for msg in messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if prompt := st.chat_input("Type your message..."):
    messages.append({"role": "user", "content": prompt})
    if active_chat["title"] == "New Chat":
        active_chat["title"] = prompt[:30] + ("..." if len(prompt) > 30 else "")

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        partial = ""
        start_time = time.time()

        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            partial += delta
            placeholder.markdown(partial + "▌")

        placeholder.markdown(partial)

        elapsed = time.time() - start_time
        word_count = len(partial.split())
        st.caption(f"{word_count} words · {elapsed:.1f}s")

    messages.append({"role": "assistant", "content": partial})
    st.rerun()
