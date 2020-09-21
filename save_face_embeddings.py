import cv2
import argparse
import os
import numpy as np
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input


# extract face and calculate face embeddings in a photo
def get_embeddings(filepath):
    face = cv2.imread(filepath)
    image = Image.fromarray(face)
    face_array = asarray(image)
    samples = asarray([face_array], 'float32')
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples, version=2)
    # create a vggface model
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    # perform prediction
    yhat = model.predict(samples)
    return yhat

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image')
    parser.add_argument('--destination_dir')
    args = parser.parse_args()
    return args



