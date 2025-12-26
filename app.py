import streamlit as st
import numpy as np
from PIL import Image
import cv2
from firebase_utils import db, bucket
from face_processor import load_model, get_embedding

# Face quality checks
def is_low_light(img_np, threshold=50):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    return np.mean(gray) < threshold

def is_blurry(img_np, threshold=110):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


st.set_page_config("Face Registration", layout="centered")
st.title("📸 Student Face Registration")

model = load_model()

POSES = ["Front", "Left", "Right", "Up", "Down"]

# Session init
if "step" not in st.session_state: st.session_state.step = 0
if "faces" not in st.session_state: st.session_state.faces = []


# ------------------------ CLASS SELECTION ------------------------
st.subheader("1. Select Class")

col1, col2 = st.columns(2)
dept = col1.selectbox("Department", ["CSE","ISE","AI/ML","CS-ML","CS-DS","AI/DS","MBA","MCA"])
batch = col2.text_input("Batch", placeholder="2024")
section = st.text_input("Section", placeholder="A")

if not (dept and batch and section):
    st.stop()

class_id = f"{dept}_{batch}_{section}"

# Fetch students
students_ref = db.collection("classes").document(class_id).collection("students").stream()
students = {s.id:s.to_dict() for s in students_ref}

if not students:
    st.warning("⚠ No student list exists for this class. Ask admin to upload CSV.")
    st.stop()

st.subheader("2. Select Your USN")
usn = st.selectbox("USN", list(students.keys()))
student = students[usn]

st.info(f"👤 Name: **{student['name']}**")

if student.get("face_registered"):
    st.error("Face already registered. Contact admin to update.")
    st.stop()


# ------------------------ POSE CAPTURE ------------------------
st.subheader("3. Capture Face Poses")

current_pose = POSES[st.session_state.step]
st.markdown(f"### Capture Pose: **{current_pose}** ({st.session_state.step+1}/5)")
st.markdown("📌 Ensure only one face is visible in correct pose.")

img_file = st.camera_input("Capture Photo", key=current_pose)

if img_file and st.button("Save This Pose"):
    img = Image.open(img_file).convert("RGB")
    img_np = np.array(img)

    # Quality checks
    if is_low_light(img_np): st.error("Too dark! Move to better light."); st.stop()
    if is_blurry(img_np): st.error("Image blurry. Hold steady."); st.stop()

    embedding = get_embedding(model, img_np)
    if embedding is None:
        st.error("No clear face detected. Try again.")
        st.stop()

    st.session_state.faces.append(embedding)
    st.session_state.step += 1
    st.success(f"{current_pose} captured ✔")
    st.rerun()


# ------------------------ FINAL SUBMISSION ------------------------
if st.session_state.step == 5:
    st.success("🎉 All 5 poses captured!")

    if st.button("Finalize Registration"):
        final_embedding = np.mean(st.session_state.faces, axis=0).tolist()

        # Save final embedding
        db.collection("classes").document(class_id) \
          .collection("students").document(usn).update({
              "embedding": final_embedding,
              "face_registered": True
          })

        # Save one face image preview (optional)
        bucket.blob(f"faces/{class_id}/{usn}.jpg")\
              .upload_from_string(img_file.read(), content_type="image/jpeg")

        st.balloons()
        st.success(f"Face registered for **{student['name']} ({usn})**")

        # Reset
        st.session_state.step = 0
        st.session_state.faces = []
