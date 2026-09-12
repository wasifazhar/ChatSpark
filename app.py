import os
import streamlit as st
from groq import Groq

st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("AI Chatbot")

api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
client = Groq(api_key=api_key)

MODEL_OPTIONS = {
    "GPT-OSS 20B (fast)": "openai/gpt-oss-20b",
    "GPT-OSS 120B (smarter)": "openai/gpt-oss-120b",
}

with st.sidebar:
    st.subheader("Settings")
    selected_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()))
    MODEL = MODEL_OPTIONS[selected_label]
    if st.button("Clear chat"):
        st.session_state.messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        partial = ""

        stream = client.chat.completions.create(
            model=MODEL,
            messages=st.session_state.messages,
            temperature=0.7,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            partial += delta
            placeholder.markdown(partial + "▌")

        placeholder.markdown(partial)

    st.session_state.messages.append({"role": "assistant", "content": partial})
