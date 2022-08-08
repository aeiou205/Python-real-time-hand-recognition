#librerias-------------------------------------------------------------
from typing import Any
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from google.protobuf.json_format import MessageToDict

from dataclasses import dataclass

@dataclass(frozen=True)
class mp_factory:
    mpHands : Any = mp.solutions.hands
    hands : Any  = mpHands.Hands(max_num_hands=4, min_detection_confidence=0.7)
    mpDraw : Any = mp.solutions.drawing_utils
    mp_drawing_styles : Any = mp.solutions.drawing_styles

@dataclass
class tf_factory:
    classNames : Any 
    model : Any

    def __init__(self):
        self.load_model()

    def load_model(self):
        self.model = load_model('mp_hand_gesture')
        f = open('gesture.names', 'r')
        self.classNames = f.read().split('\n')
        f.close()


#---------------------------------------------------------------------
def right_left_hand_detection( result_hands_process, canvas ):

    if result_hands_process.multi_hand_landmarks:
        #---------------------Both Hands'-------
        if len(result_hands_process.multi_handedness) == 2:
            cv2.putText(canvas, 'Both Hands', (250, 50),cv2.FONT_HERSHEY_COMPLEX,0.9, (0, 255, 0), 2)
        #---------------------Lefth hand--------
        else:
            for i in result_hands_process.multi_handedness:
                label = MessageToDict(i)['classification'][0]['label']
                if label == 'Left':
                    cv2.putText(canvas, label+' Hand',(20, 50),cv2.FONT_HERSHEY_COMPLEX,0.9, (0, 255, 0), 2)
        #---------------------Rigth Hand--------
                if label == 'Right':
                    cv2.putText(canvas, label+' Hand',(320, 50),cv2.FONT_HERSHEY_COMPLEX,0.9, (0, 255, 0), 2)

#--------------------------------------------------------------------
def detected_hands_keypoints(result_hands_process, mp_vars, tf_vars, canvas):

    if result_hands_process.multi_hand_landmarks:
        x,y,_ = canvas.shape
        point_medium = []
        landmarks = []
        for handslms in result_hands_process.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])

            x1 = int(handslms.landmark[mp_vars.mpHands.HandLandmark.THUMB_TIP].x * y)
            y1 = int(handslms.landmark[mp_vars.mpHands.HandLandmark.THUMB_TIP].y * x)

            x2 = int(handslms.landmark[mp_vars.mpHands.HandLandmark.INDEX_FINGER_TIP].x * y)
            y2 = int(handslms.landmark[mp_vars.mpHands.HandLandmark.INDEX_FINGER_TIP].y * x)

            point_medium.append( ( (x1+x2)//2, (y1+y2)//2) )
            
            mp_vars.mpDraw.draw_landmarks(canvas, handslms, mp_vars.mpHands.HAND_CONNECTIONS,
            mp_vars.mp_drawing_styles.get_default_hand_landmarks_style(),mp_vars.mp_drawing_styles.get_default_hand_connections_style())   
    
        # tuple(axesMayor, axe)
        # if len(point_medium)>1 :
        #cv2.ellipse(canvas,point_medium[0],(100,50),30,0,360,(0, 0, 0),thickness=5)
        cv2.circle(canvas, point_medium[0], 3,(0, 0, 0),25)
        
        if len(point_medium)>1 : 
            cv2.circle(canvas, point_medium[1], 3,(0, 0, 0),25)
            cv2.line(canvas,point_medium[0],point_medium[1],(55, 255, 100),thickness=10)
            # cv2.ellipse(canvas,point_medium[0],point_medium[1],(100,50),30,0,360,(0, 0, 0),thickness=5)
            cv2.ellipse(canvas,point_medium[0],(50,50),30,0,360,(0, 0, 0),thickness=5)
            cv2.ellipse(canvas,point_medium[1],(50,50),30,0,360,(0, 0, 0),thickness=5)
            
        prediction = tf_vars.model.predict([landmarks])#<-----------------yo
        print("here: ", prediction)
        print(len(prediction))
        pred = 0
        for i, pred in enumerate(prediction):
            classID = np.argmax(pred)#<-----------------yo
            print(classID)
            className = tf_vars.classNames[classID] #<----------------yo
            print("className : ", className )
            cv2.putText(canvas, className, (300+i*100, 30),  cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2, cv2.LINE_AA) 


def main():

    mp_vars = mp_factory()
    tf_vars = tf_factory() 
    print(tf_vars.classNames)

    cap = cv2.VideoCapture(0)
    while True:

        _, frame = cap.read()
        x, y, _ = frame.shape
        # Flip the frame vertically
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Get hand landmark prediction
        result_hands_process = mp_vars.hands.process(framergb)
        # print(result)
        ##print(frame)
        className = ''
        right_left_hand_detection(result_hands_process, frame )
        detected_hands_keypoints(result_hands_process, mp_vars, tf_vars, frame)


        cv2.imshow("Output", frame) 
        if cv2.waitKey(1) == ord('q'):
            break
    # release the webcam and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
