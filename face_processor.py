import numpy as np
from insightface.app import FaceAnalysis

def load_model():
    model = FaceAnalysis(name="buffalo_l")
    model.prepare(ctx_id=0, det_size=(640, 640))
    return model

def get_embedding(model, img):
    faces = model.get(img)
    return faces[0].embedding if len(faces)==1 else None
