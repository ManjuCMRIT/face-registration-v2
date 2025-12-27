import streamlit as st
import numpy as np
from PIL import Image
import cv2

from firebase_utils import db, bucket
from face_processor import load_model, get_embedding

# ===================== SETTINGS =====================
POSES = ["Front", "Left", "Right", "Up", "Down"]

# UI Config
st.set_page_config("Face Registration", layout="centered")
st.title("📸 Student Face Registration")


# ===================== LOAD MODEL (Cached) =====================
@st.cache_resource
def load_face_model():
    return load_model()

model = load_face_model()


# ===================== QUALITY CHECKS =====================
def is_low_light(img_np, threshold=60):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    return np.mean(gray) < threshold

def is_blurry(img_np, threshold=110):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold



# ===================== SESSION INIT =====================
if "step" not in st.session_state:
    st.session_state.step = 0
if "faces" not in st.session_state:
    st.session_state.faces = []
if "last_image" not in st.session_state:
    st.session_state.last_image = None


# ===================== CLASS SELECTION =====================
st.subheader("1. Select Class")

col1, col2 = st.columns(2)
dept = col1.selectbox("Department", ["CSE","ISE","AI/ML","CS-ML","CS-DS","AI/DS","MBA","MCA"])
batch = col2.text_input("Batch", placeholder="2024")
section = st.text_input("Section", placeholder="A")

if not (dept and batch and section):
    st.stop()

class_id = f"{dept}_{batch}_{section}"

# Load only USN & name (light weight)
student_docs = db.collection("classes").document(class_id).collection("students").stream()
students = {s.id: s.to_dict()['name'] for s in student_docs}

if not students:
    st.error("⚠ No student list found. Admin must upload CSV first.")
    st.stop()


# ===================== SELECT STUDENT =====================
st.subheader("2. Select USN")
usn = st.selectbox("USN", list(students.keys()))
student = db.collection("classes").document(class_id).collection("students").document(usn).get().to_dict()

st.info(f"👤 Name: **{student['name']}**")

if student.get("face_registered"):
    st.error("Face already registered. Contact admin to update.")
    st.stop()


# ===================== CAPTURE POSES =====================
st.subheader("3. Capture 5 Face Poses")

# While poses are remaining
if st.session_state.step < len(POSES):
    pose = POSES[st.session_state.step]
    st.markdown(f"### Capture Pose: **{pose}** ({st.session_state.step+1}/5)")

    img_file = st.camera_input(f"Capture {pose}")

    if img_file and st.button("Save This Pose"):
        img = Image.open(img_file).convert("RGB")
        img_np = np.array(img)

        # Quality checks
        if is_low_light(img_np):
            st.error("⚠ Low light. Move to brighter location."); st.stop()

        if is_blurry(img_np):
            st.error("⚠ Image blurry. Hold still."); st.stop()

        embedding = get_embedding(model, img_np)
        if embedding is None:
            st.error("No face detected. Retake image."); st.stop()

        st.session_state.faces.append(embedding)
        st.session_state.last_image = img_file   # store last captured image
        st.session_state.step += 1
        st.success(f"{pose} captured ✓")
        st.rerun()


# ===================== FINAL SUBMISSION =====================
if st.session_state.step == 5:
    st.success("🎉 All 5 poses captured successfully!")

    if st.button("Finalize Registration 🚀"):
        final_embedding = np.mean(st.session_state.faces, axis=0).tolist()

        # Save to database
        db.collection("classes").document(class_id).collection("students").document(usn).update({
            "embedding": final_embedding,
            "face_registered": True
        })

        # Upload face preview image to storage
        if st.session_state.last_image:
            blob = bucket.blob(f"faces/{class_id}/{usn}.jpg")
            blob.upload_from_string(st.session_state.last_image.read(), content_type="image/jpeg")

        st.balloons()
        st.success(f"Face Registered for **{student['name']} ({usn})**")

        # RESET for next student
        st.session_state.step = 0
        st.session_state.faces = []
        st.session_state.last_image = None
        st.rerun()
