import streamlit as st
import requests as api

import time


st.title("Drunk Bot (:")

status = api.get("http://127.0.0.1:8000/")
status = status.json()
st.badge(status.get("message"))

#initialize of chat history
if "chats" not in st.session_state:
    st.session_state.chats = []

#chat box
chat_box = st.container(height=400)

for chat in st.session_state.chats:
    if chat.get("role") == "user":
        chat_box.chat_message("user").write(f"user: {chat['content']}")
    else:   
        chat_box.chat_message("bot").write(f"bot: {chat['content']}")

#taking input
prompt = st.chat_input("Say Hi!")


if prompt:
    st.session_state.chats.append({"role":"user", "content":prompt})

    chat_box.chat_message("user").write(f"user: {prompt}")
    data = {}
    data["text"] = prompt
    try:
        response = api.post("http://127.0.0.1:8000/chat", json=data)
        response = response.json()
        st.session_state.chats.append({"role":"bot", "content":response.get('output')})
        chat_box.chat_message("bot").write(f"bot: {response.get('output')}")
        
    except Exception as e:
        print(e)