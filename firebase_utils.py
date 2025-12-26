import firebase_admin
from firebase_admin import credentials, firestore, storage
import json
import streamlit as st

if not firebase_admin._apps:
    cred = credentials.Certificate(
        json.loads(st.secrets["FIREBASE_KEY"])
    )
    firebase_admin.initialize_app(
        cred,
        {"storageBucket": "face-registration-app.firebasestorage.app"}
    )

db = firestore.client()
bucket = storage.bucket()
