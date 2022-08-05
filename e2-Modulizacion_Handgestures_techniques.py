#librerias-------------------------------------------------------------
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from google.protobuf.json_format import MessageToDict
landmarks = []

#---------------------------------------------------------------------
def initialize_mediaPipe():
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,model_complexity=1,min_detection_confidence=0.75,
        min_tracking_confidence=0.75,
        max_num_hands=2)
    return (mpHands,hands,mpDraw,mp_drawing_styles)

#--------------------------------------------------------------------
def initialize_tensorflow():
    model = load_model('mp_hand_gesture')#Load the gesture recognizer model
    # Load class names
    f = open('gesture.names', 'r')
    classNames = f.read().split('\n')
    f.close()
    #print(classNames)
    return(classNames)

#---------------------------------------------------------------------
def right_left_hand_detection():
    if result.multi_hand_landmarks:
        #---------------------Both Hands'-------
        if len(result.multi_handedness) == 2:
            cv2.putText(frame, 'Both Hands', (250, 50),cv2.FONT_HERSHEY_COMPLEX,0.9, (0, 255, 0), 2)
        #---------------------Lefth hand--------
        else:
            for i in result.multi_handedness:
                label = MessageToDict(i)['classification'][0]['label']
                if label == 'Left':
                    cv2.putText(frame, label+' Hand',(20, 50),cv2.FONT_HERSHEY_COMPLEX,0.9, (0, 255, 0), 2)
        #---------------------Rigth Hand--------
                if label == 'Right':
                    cv2.putText(frame, label+' Hand',(20, 50),cv2.FONT_HERSHEY_COMPLEX,0.9, (0, 255, 0), 2)


def detected_hands_keypoints(mpHands,hands,mpDraw,mp_drawing_styles):
    if result.multi_hand_landmarks:
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])
            x1 = int(handslms.landmark[mpHands.HandLandmark.THUMB_TIP].x * y)
            y1 = int(handslms.landmark[mpHands.HandLandmark.THUMB_TIP].y * x)
            point1 = ((x1+y1)/2) 
            p=point1

            x2 = int(handslms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x * y)
            y2 = int(handslms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y * x)
            point2 = ((x2+y2)/2)
            p2=point2

            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),mp_drawing_styles.get_default_hand_connections_style())   
            cv2.circle(frame, (x1,y1), 3,(255, 0, 0),10)
            cv2.circle(frame, (x2,y2), 3,(255, 0, 0),10)



mpHands,hands,mpDraw,mp_drawing_styles=initialize_mediaPipe()
classNames=initialize_tensorflow()
print(classNames)
landmarks = []
print(landmarks)


cap = cv2.VideoCapture(1)
while True:


    _, frame = cap.read()
    x, y, _ = frame.shape
    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Get hand landmark prediction
    result = hands.process(framergb)
    # print(result)
    ##print(frame)
    className = ''
    right_left_hand_detection()
    detected_hands_keypoints(mpHands,hands,mpDraw,mp_drawing_styles)


    cv2.imshow("Output", frame) 
    if cv2.waitKey(1) == ord('q'):
        break
# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
        
