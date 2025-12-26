import streamlit as st
import numpy as np
from PIL import Image
import cv2
from firebase_utils import db, bucket
from face_processor import load_model, get_embedding

st.set_page_config("Student Face Registration", layout="centered")
st.title("📸 Student Face Registration Portal")

model = load_model()

# ----------------- Select Class -----------------
dept = st.selectbox("Select Department", ["CSE","ISE","AI/ML","CS-ML","CS-DS","AI/DS","MBA","MCA"])
batch = st.text_input("Batch (Year of Admission)")
section = st.text_input("Section")

if dept and batch and section:
    class_id = f"{dept}_{batch}_{section}"

    students_ref = db.collection("classes").document(class_id).collection("students").stream()
    students = {s.id: s.to_dict() for s in students_ref}

    if students:
        usn = st.selectbox("Select USN", list(students.keys()))
        st.write("Name:", students[usn]["name"])
    else:
        st.warning("No students found for this class. Upload CSV first in Admin Panel.")
        st.stop()

# ----------------- Camera Capture -----------------
img_file = st.camera_input("Capture Face")

if img_file and st.button("Register Face"):
    if students[usn]["face_registered"]:
        st.error("Face already registered for this student.")
        st.stop()

    img = Image.open(img_file).convert("RGB")
    img_np = np.array(img)

    embedding = get_embedding(model, img_np)

    if embedding is None:
        st.error("Face not detected properly. Ensure only ONE face is visible.")
        st.stop()

    # Save image to Firebase Storage
    blob = bucket.blob(f"faces/{class_id}/{usn}.jpg")
    blob.upload_from_string(img_file.read(), content_type="image/jpeg")

    # Save embedding to Firestore
    db.collection("classes").document(class_id) \
      .collection("students").document(usn).update({
          "embedding": embedding.tolist(),
          "face_registered": True
      })

    st.success(f"Face successfully registered for {usn}")
