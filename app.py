import streamlit as st
import numpy as np
from PIL import Image
import cv2
from io import BytesIO

from firebase_utils import db, bucket
from face_processor import load_model, get_embedding

# ---------------- Quality Checks ----------------
def is_low_light(img_np, threshold=50):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    return np.mean(gray) < threshold

def is_blurry(img_np, threshold=110):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


# ---------------- App Setup ----------------
st.set_page_config("Face Registration", layout="centered")
st.title("📸 Student Face Registration Portal")

model = load_model()
POSES = ["Front", "Left", "Right", "Up", "Down"]


# ---------------- Session ----------------
if "step" not in st.session_state: st.session_state.step = 0
if "embeds" not in st.session_state: st.session_state.embeds = []
if "images" not in st.session_state: st.session_state.images = []


# ---------------- Class Selection ----------------
st.subheader("1️⃣ Select Class")

col1, col2 = st.columns(2)
dept = col1.selectbox("Department", ["CSE","ISE","AI/ML","CS-ML","CS-DS","AI/DS","MBA","MCA"])
batch = col2.text_input("Batch", placeholder="2024")
section = st.text_input("Section", placeholder="A")

if not (dept and batch and section):
    st.stop()

class_id = f"{dept}_{batch}_{section}"

students_ref = db.collection("classes").document(class_id).collection("students").stream()
students = {s.id: s.to_dict() for s in students_ref}

if not students:
    st.error("⚠ No student list found. Ask admin to upload CSV.")
    st.stop()


# ---------------- Student Selection ----------------
st.subheader("2️⃣ Select Your USN")
usn = st.selectbox("Select USN", list(students.keys()))
student = students[usn]

st.info(f"👤 **{student['name']}** — {usn}")

if student.get("face_registered"):
    st.warning("⚠ Face already registered. Contact admin to update.")
    st.stop()


# ---------------- Pose Capture ----------------
if st.session_state.step < 5:

    pose = POSES[st.session_state.step]
    st.subheader(f"3️⃣ Capture Pose — **{pose}** ({st.session_state.step+1}/5)")
    st.write("📌 Take photo in good light. Keep face centered.")

    file = st.camera_input(f"Capture {pose}")

    if file and st.button("Save Pose"):
        img = Image.open(file).convert("RGB")
        img_np = np.array(img)

        # quality
        if is_low_light(img_np):
            st.error("Too dark ❌ Try brighter light")
            st.stop()
        if is_blurry(img_np):
            st.error("Image blurry ❌ Hold still")
            st.stop()

        embed = get_embedding(model, img_np)
        if embed is None:
            st.error("Face not detected ❌ Retake")
            st.stop()

        st.session_state.embeds.append(embed)
        st.session_state.images.append(img)
        st.session_state.step += 1
        st.success("Pose saved ✔")
        st.rerun()


# ---------------- Final Save ----------------
if st.session_state.step == 5:
    st.success("🎉 All 5 poses captured!")

    if st.button("Finalize Registration 🚀"):

        # average embedding
        final_embed = np.mean(st.session_state.embeds, axis=0).tolist()

        # store embedding
        db.collection("classes").document(class_id)\
            .collection("students").document(usn).update({
                "embedding": final_embed,
                "face_registered": True
            })

        # -------- upload face images properly --------
        for i, img in enumerate(st.session_state.images):
            buf = BytesIO()
            img.save(buf, format="JPEG")
            buf.seek(0)

            blob = bucket.blob(f"faces/{class_id}/{usn}_{POSES[i]}.jpg")
            blob.upload_from_file(buf, content_type="image/jpeg")

        st.balloons()
        st.success(f"✔ Registration Completed for {student['name']}")

        # reset for next user
        st.session_state.step = 0
        st.session_state.embeds = []
        st.session_state.images = []
        st.rerun()
