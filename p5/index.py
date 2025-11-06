import streamlit as st
import requests as api
import time

st.title("Brain Tumor Detection")
st.subheader("ConvNeXT Tiny Based", divider=True)
file=st.file_uploader("file")

stat = api.get("http://127.0.0.1:8000/")

if file is not None:
    prog_bar = st.progress(0, text="loading file...")
    st.spinner()

    for load_percent in range(100):
        time.sleep(0.01)
        prog_bar.progress(load_percent+1, text="loading file...")

    st.image(file)
    prog_bar.empty()
   
    files = {"file" : (file.name, file, file.type)}
   
    try:
        response = api.post("http://127.0.0.1:8000/predict", files=files)
        data = response.json()
        st.subheader("Result : "+ data.get("prediction"))
        response_time = round(data.get("response"), 2)
        st.badge("Response Time : " + str(response_time)+" ms")
    except Exception as e:
        st.warning(e)
    