import cv2
import os
import argparse
import numpy as np
from dface.core.detect import create_mtcnn_net, MtcnnDetector

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/sahar/Documents/Mahmoudi_Benchohra_Project/Speaker_Verification_Vox1/')
from hparam import hparam as hp



def extract_face(img, bboxs, landmarks):

    if img is None:
        return "Image is None"
    if len(bboxs)==0 or len(landmarks)==0:
        return "No faces detected"

    main_face = 0
    main_face_w = 0
    main_face_h = 0

    for i in range(bboxs.shape[0]):
        bbox = bboxs[i, :4]
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        width = x2-x1
        height = y2-y1

        if (width*height > main_face_w*main_face_h) :
            main_face = i
            main_face_w = width
            main_face_h = height

    bbox = bboxs[main_face, :4]
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    center = ( (x1+x2)/2 , (y1+y2)/2 )
    width = int(x2-x1)
    height = int(y2-y1)

    landmarks = landmarks[main_face, :4]
    left_eye_x = landmarks[0]
    left_eye_y = landmarks[1]
    right_eye_x = landmarks[2]
    right_eye_y = landmarks[3]
    angle = 180/np.pi*np.arctan((left_eye_y - right_eye_y) / (right_eye_x - left_eye_x))

    # Only correct the rotation if the angle is more than 10 degrees.
    if (np.abs(angle) > 10):
        print("Face is at an angle of %.0f degrees, correcting ..." % angle)
        extracted_face = rotate_and_crop_image(img, center, -angle, width, height)
    else:
        extracted_face = img[ int(y1):int(y2) , int(x1):int(x2) ]

    return extracted_face


def rotate_and_crop_image(image, center, theta, width, height):
   ''' 
   Rotates OpenCV image around center with angle theta (in deg)
   then crops the image according to width and height.
   '''

   shape = ( image.shape[1], image.shape[0] ) # cv2.warpAffine expects shape in (length, height)

   matrix = cv2.getRotationMatrix2D( center=center, angle=theta, scale=1 )
   image = cv2.warpAffine( src=image, M=matrix, dsize=shape )

   x = int( center[0] - width/2 )
   y = int( center[1] - height/2 )

   image = image[ y:y+height, x:x+width ]

   return image



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image')
    parser.add_argument('--destination_dir')
    args = parser.parse_args()
    return args



if __name__ == '__main__':

    args = parse_args()

   # pnet, rnet, onet = create_mtcnn_net(p_model_path="./model_store/pnet_epoch.pt", r_model_path="./model_store/rnet_epoch.pt", o_model_path="./model_store/onet_epoch.pt", use_cuda=False)
    pnet, rnet, onet = create_mtcnn_net(p_model_path=os.path.dirname(os.path.abspath(__file__))+"/model_store/pnet_epoch.pt"
                                      , r_model_path=os.path.dirname(os.path.abspath(__file__))+"/model_store/rnet_epoch.pt"
                                      , o_model_path=os.path.dirname(os.path.abspath(__file__))+"/model_store/onet_epoch.pt", use_cuda=False) 
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)

    img = cv2.imread(args.input_image)

    bboxs, landmarks = mtcnn_detector.detect_face(img)
    face = extract_face(img, bboxs, landmarks)
    if isinstance(face, str):
        print(face)
        raise SystemExit

    face = cv2.resize(face, (224,224))

    saving_name = os.path.basename(args.input_image).replace(".jpg", "_visage.jpg")
    saving_path = os.path.join(args.destination_dir, saving_name)
    cv2.imwrite(saving_path, face)

