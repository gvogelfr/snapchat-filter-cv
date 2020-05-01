import numpy as np
import cv2
import dlib
import time
import sys
from imutils import face_utils
import imutils

def run():
    cap = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    while True:
        if len(sys.argv)>1:
            image = cv2.imread(sys.argv[1])
        else:
            ret, image = cap.read()
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray,1)
        
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            #left ear 
            left_ear_image = cv2.imread("mondrian_left.png", -1)
            orig_mask_l = left_ear_image[:,:,3]
            orig_mask_inv_l = cv2.bitwise_not(orig_mask_l)
            left_ear_image = left_ear_image[:,:,0:3]
            original_left_height, original_left_width = left_ear_image.shape[:2]
            place_ear(image, original_left_width, original_left_height, 
            shape, left_ear_image, orig_mask_l, orig_mask_inv_l, True)

            #right ear
            right_ear_image = cv2.imread("mondrian_right.png", -1)
            orig_mask_r = right_ear_image[:,:,3]
            orig_mask_inv_r = cv2.bitwise_not(orig_mask_r)
            right_ear_image = right_ear_image[:,:,0:3]
            original_right_height, original_right_width = right_ear_image.shape[:2]
            place_ear(image, original_right_width, original_right_height, shape, right_ear_image, orig_mask_r, orig_mask_inv_r, False)
        cv2.imshow("Output", image)
        k = cv2.waitKey(30) & 0xff
        if k == 27: # press 'ESC' to quit
            break
    cap.release()
    cv2.destroyAllWindows()


def place_ear(frame, ear_width, ear_height, shape, image, orig_mask, orig_mask_inv, left):  
    if left:
        width = int(1.3*(np.linalg.norm(shape[0] - shape[19])))
        height = int(width * ear_height / ear_width)
        x1 = int(shape[0,0] - (width/2))
        x2 = int(x1 + width)  
        y1 = int(shape[0,1] -(height/2)) -70
        y2 = int(y1 + height)
    else:
        width = int(1.3*(np.linalg.norm(shape[24] - shape[16])))
        height = int(width * ear_height / ear_width)
        x1 = int(shape[16,0] - (width/2)) 
        x2 = int(x1 + width)  
        y1 = int(shape[16,1] -(height/2)) -70
        y2 = int(y1 + height)

    h, w = frame.shape[:2] 
 
    if not (x1 < 0 or y1 <0 or x2>w or y2>h):  
        earOverlayWidth = x2 - x1  
        earOverlayHeight = y2 - y1  

        
        earOverlay = cv2.resize(image, (earOverlayWidth,earOverlayHeight), interpolation = cv2.INTER_AREA)  
        mask = cv2.resize(orig_mask, (earOverlayWidth,earOverlayHeight), interpolation = cv2.INTER_AREA)  
        mask_inv = cv2.resize(orig_mask_inv, (earOverlayWidth,earOverlayHeight), interpolation = cv2.INTER_AREA) 

        roi = frame[y1:y2, x1:x2]  
        roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)  
        roi_fg = cv2.bitwise_and(earOverlay,earOverlay,mask = mask)  
        dst = cv2.add(roi_bg,roi_fg) 
        frame[y1:y2, x1:x2] = dst 
run()