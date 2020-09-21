import cv2
import argparse
import os
import time
import pickle
import numpy as np
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/sahar/Documents/Mahmoudi_Benchohra_Project/Speaker_Verification_Vox1/')
sys.path.insert(1, '/home/sahar/benchohra/DFace_production/')

from hparam import hparam as hp

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
  

# compare two embeddings and return the distance
def compare(known_embedding, candidate_embedding):
    # calculate distance between embeddings
    score = cosine(known_embedding, candidate_embedding)
    return score



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--face_image', dest='face_image', type=str)
    parser.add_argument('--preprocessed_dir')
    parser.add_argument('--best_identified_faces')
    parser.add_argument('--best_identified_speakers', default="NONE")
    args = parser.parse_args()
    return args


""" # ENROLL ========================================================================================================================================================

"conda run -n pytorch_main python " + hp.integration.face_verification_path + "extract_face.py --input_image " + img_file_path + " --destination_dir " + hp.integration.enroll_preprocessed_photo

# VERIFY ========================================================================================================================================================
 
"conda run -n pytorch_main python " + hp.integration.face_verification_path + "extract_face.py --input_image " + img_file_path + " --destination_dir " + hp.integration.verify_upload_folder

"conda run -n vgg_py3 python " + hp.integration.face_verification_path + "identify_face.py --face_image " + os.path.splitext(img_file_path)[0]+"_visage.jpg" + " --preprocessed_dir " + hp.integration.enroll_preprocessed_photo

# =============================================================================================================================================================== """


if __name__ == '__main__':

    args = parse_args()

    """ if args.best_identified_speakers != "NONE":
        with open(args.best_identified_faces + 'speaker_result.data', 'rb') as filehandler:
            best_identified_speakers = pickle.load(filehandler)
        restriction_list = [x[0] for x in best_identified_speakers] """

    accuracy_list = []

    start_em_f = time.time()
    embeddings_face = get_embeddings(args.face_image)
    print("Got face embeddings in : %f" % (time.time() - start_em_f))

    start_loop = time.time()
    for face_file in os.listdir(args.preprocessed_dir):
        if face_file[-4:]=='.jpg':
            continue
        print("Loop for (%s)" % (face_file))
        id = '_'.join(face_file.split('_')[:4])

        # Check if a list of best speakers is provided
        # If so : only check against the list provided by the speech identification
        # Else (list not provided) : check against all database
        """ if (args.best_identified_speakers == "NONE") or (id in restriction_list):
            face_path = os.path.join(args.preprocessed_dir, face_file)
            start_em_b = time.time()
            embeddings_base = get_embeddings(face_path)
            print("Got base embeddings in : %f" % (time.time() - start_em_b))
            start_comp = time.time()
            score = compare(embeddings_face, embeddings_base)
            print("Done comparing in : %f" % (time.time() - start_comp))
            score = round(score, 2)
            accuracy = 1-score
            accuracy_list.append((id, accuracy)) """

        """ face_path = os.path.join(args.preprocessed_dir, face_file)
        start_em_b = time.time()
        embeddings_base = get_embeddings(face_path)
        print("Got base embeddings in : %f" % (time.time() - start_em_b))
        start_comp = time.time()
        score = compare(embeddings_face, embeddings_base)
        print("Done comparing in : %f" % (time.time() - start_comp))
        score = round(score, 2)
        accuracy = 1-score
        accuracy_list.append((id, accuracy)) """



    print("Done with the loop in : %f" % (time.time() - start_loop))

    accuracy_list.sort(key=lambda tup: tup[1], reverse=True)
    with open(args.best_identified_faces+'facial_result.data', 'wb') as f:
        pickle.dump(accuracy_list, f)

