import streamlit as st 
import numpy as np
from PIL import Image
import cv2

from firebase_utils import db, bucket
from face_processor import load_model, get_embedding


# ------------------------ Quality checks ------------------------
def is_low_light(img_np, threshold=50):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    return np.mean(gray) < threshold

def is_blurry(img_np, threshold=110):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


# ------------------------ App Setup ------------------------
st.set_page_config("Face Registration", layout="centered")
st.title("📸 Student Face Registration")

model = load_model()
POSES = ["Front", "Left", "Right", "Up", "Down"]


# ------------------------ Session State ------------------------
if "step" not in st.session_state:
    st.session_state.step = 0

if "faces" not in st.session_state:
    st.session_state.faces = []


# ------------------------ CLASS SELECTION ------------------------
st.subheader("1. Select Class")

col1, col2 = st.columns(2)
dept = col1.selectbox("Department", ["CSE","ISE","AI/ML","CS-ML","CS-DS","AI/DS","MBA","MCA"])
batch = col2.text_input("Batch", placeholder="2024")
section = st.text_input("Section", placeholder="A")

if not (dept and batch and section):
    st.stop()

class_id = f"{dept}_{batch}_{section}"

students_ref = db.collection("classes").document(class_id).collection("students").stream()
students = {s.id:s.to_dict() for s in students_ref}

if not students:
    st.warning("⚠ No student list found. Ask admin to upload CSV first.")
    st.stop()


# ------------------------ STUDENT SELECTION ------------------------
st.subheader("2. Select Your USN")
usn = st.selectbox("USN", list(students.keys()))
student = students[usn]

st.info(f"👤 Name: **{student['name']}**")

if student.get("face_registered"):
    st.error("❌ Face already registered. Contact admin to update.")
    st.stop()


# ------------------------ POSE CAPTURE ------------------------
st.subheader("3. Capture 5 Face Poses")

if st.session_state.step < len(POSES):

    current_pose = POSES[st.session_state.step]
    st.markdown(f"### Capture Pose: **{current_pose}** ({st.session_state.step+1}/5)")
    st.markdown("📌 Ensure only one face is visible. Good lighting recommended.")

    img_file = st.camera_input(f"Capture {current_pose} Pose")

    if img_file and st.button("Save This Pose"):
        img = Image.open(img_file).convert("RGB")
        img_np = np.array(img)

        if is_low_light(img_np): 
            st.error("⚠ Too dark. Move to better light.")
            st.stop()

        if is_blurry(img_np):
            st.error("⚠ Image is blurry. Try again.")
            st.stop()

        embedding = get_embedding(model, img_np)
        if embedding is None:
            st.error("❌ No face detected. Try again.")
            st.stop()

        st.session_state.faces.append(embedding)
        st.session_state.step += 1
        st.success(f"✔ {current_pose} captured")
        st.rerun()


# ------------------------ FINAL SUBMISSION ------------------------
elif st.session_state.step == 5:
    st.success("🎉 All 5 poses captured successfully!")

    if st.button("Finalize Registration 🚀"):
        final_embedding = np.mean(st.session_state.faces, axis=0).tolist()

        db.collection("classes").document(class_id)\
          .collection("students").document(usn).update({
              "embedding": final_embedding,
              "face_registered": True
          })

        st.success(f"🎉 Face Registered for **{student['name']} ({usn})**")
        st.balloons()

        # Reset state for next student
        st.session_state.step = 0
        st.session_state.faces = []
        st.rerun()
