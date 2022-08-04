import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from google.protobuf.json_format import MessageToDict

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initializing the Model
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,model_complexity=1,min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2)


#--------------------------------yo
# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)
#--------------------------------

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    x, y, _ = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)#(yo)<-----------------------------------------------
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # print(result)
    ##print(frame)
    className = ''
    
    # post process the result

    if result.multi_hand_landmarks:
        #-----------------Both Hands'-------------------------------------------------------------
        if len(result.multi_handedness) == 2:

            # Display 'Both Hands' on the image
            cv2.putText(frame, 'Both Hands', (250, 50),cv2.FONT_HERSHEY_COMPLEX,0.9, (0, 255, 0), 2)
        #-----------------Lefth hand--------------------------------------------------------------
        else:
            for i in result.multi_handedness:
               
                # Return whether it is Right or Left Hand
                label = MessageToDict(i)['classification'][0]['label']
 
                if label == 'Left':
                   
                    # Display 'Left Hand' on
                    # left side of window
                    cv2.putText(frame, label+' Hand',
                                (20, 50),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.9, (0, 255, 0), 2)
        #------------------Rigth Hand------------------------------------------------------------
                if label == 'Right':
                    cv2.putText(frame, label+' Hand',
                                (20, 50),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.9, (0, 255, 0), 2)
        #----------------------------------------------------------------------------------------------
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])

          #-----------------------------------------------------------------------------------      
                # print("nose", x,y)
                # #system coordenates of thumb right               

                #x1 = int(handslms.landmark[mpHands.HandLandmark.THUMB_TIP].x * img_height) 
                #y1 = int(handslms.landmark[mpHands.HandLandmark.THUMB_TIP].y * img_width)
                # x2,y2 = x1+x , y1+y
                # #cordenate mano izquierda x,y
                # #y1 = int(handslms.landmark[mpHands.HandLandmark.THUMB_TIP].y * x)
                
                
                # cordenadasXI = ((x1/y)*x)
                # cordenadasYI = ((x1/y)*y)


                # print(x1,x)
                # print(y1,y)
                #system coordenates of thumb right   
                #
                x1 = int(handslms.landmark[mpHands.HandLandmark.THUMB_TIP].x * y)
                y1 = int(handslms.landmark[mpHands.HandLandmark.THUMB_TIP].y * x)
                #print("System coordates thumb TiP HAND RIGHT:",x1,y1)

                #print("cordenadas MI",x1,y1)
                # print("cordenadas MD",x2,y2)

                # distancia=(float)(x1-y1)           #x*y
                #distanciaMd=int((y1/y)*y)      #x*y


                #x11=(float)(x1(1/y))

                #y11=(float)(x1(1/x)) 
                #y2 = int(handslms.landmark[mpHands.HandLandmark.THUMB_TIP].x * x)
                #punto de referencia mano izquierda
                #print("dinstancia mano izquierda",x1)
                #print("distancia mano derecha ",y1)

                #x2 = int(handslms.landmark[mpHands.HandLandmark.THUMB_TIP].x * y) #punto de referencia mano derecha
                #y2 = int(handslms.landmark[mpHands.HandLandmark.THUMB_TIP].y * x) #punto de referencia mano izquierda
                #distancia = (float)(((x1-x2)**2 + (y1 - y2)**2)/2)
                # print("Distancia entre ambos dedos",distancia)    
                #sqrt((p1.x - p2.x)**2 + (p1.y -p2.y)**2)       
                # cv2.line(frame,(x1,0),(0,y1),(255,0,0),5)
                # #cv2.line(frame,(x,),(c,c),(225,0,0),5)
                cv2.circle(frame, (x1,y1), 3,(255, 0, 0),10)
                
                
                #cv2.line(frame, (x1, y1), (255, 255, 255),3)
                #punto1 = Punto(x1,y1)---
                #print(punto1)
                #print(calcular_distancia(x1,y1))
            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()) #dibuja las conexiones
            # mp_drawing.srwa_landamarks
            # Predict gesture
            prediction = model.predict([landmarks])#<-----------------yo
            print("here: ", prediction)
            print(len(prediction))

            for pred in prediction:
                classID = np.argmax(pred)#<-----------------yo
                print(classID)
                className = classNames[classID] #<----------------yo
                print("className : ", className )

    cv2.putText(frame, className, (300,50),  cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2, cv2.LINE_AA)    
    
    
    manoderecha = 0


    manoizquierda = 0 
    # show the prediction on the frame


    # Show the final output
    cv2.imshow("Output", frame) 
    
    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()
