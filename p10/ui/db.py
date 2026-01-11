import streamlit as st
from pymongo import MongoClient

@st.cache_resource
def get_db():
    client = MongoClient("mongodb://mongo:27017")
    return client.hiring
