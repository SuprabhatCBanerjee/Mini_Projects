import streamlit as st
import requests as api
import time

st.title("Diabetic Retinography")
st.subheader("CNN Based Analyzer", divider=True)
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
    # file_path = os.path.join("/")
    # st.write(file_path)
    files = {"file" : (file.name, file, file.type)}
   
    try:
        response = api.post("http://127.0.0.1:8000/predict/image", files=files)
        data = response.json()
        st.subheader("Result : "+ data.get("prediction"))
        response_time = round(data.get("inference_time_ms"), 2)
        st.badge("Response Time : " + str(response_time)+" ms")
    except Exception as e:
        st.warning(e)
    