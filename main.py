import itertools
import cv2
import mediapipe as mp
import numpy as np
import cvzone
from cvzone.PlotModule import LivePlot


#Constants
final_frame = 0
INITIAL_COLOR = (0,0,255) #red
B_COLOR = (0,255,255) #yellow
eyes_areas = []
counter = 0
NUMBER_OF_BLINKS = 0 #intially zero

#cvzone inits
PLOTLY = LivePlot(yLimit=[0,100], invert=True)

# meadiapipe inits
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)


cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    image_height, image_width, _ = frame.shape
    frame.flags.writeable = False
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    INDEXES_RIGHT, INDEXES_LEFT = mp_face_mesh.FACEMESH_RIGHT_EYE, mp_face_mesh.FACEMESH_LEFT_EYE
    INDEXES_LIST_RIGHT, INDEXES_LIST_LEFT = list(itertools.chain(*INDEXES_RIGHT)), list(itertools.chain(*INDEXES_LEFT))
    Rrecords, Lrecords = [], []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for INDEX in INDEXES_LIST_RIGHT:
                cv2.circle(frame,(int(face_landmarks.landmark[INDEX].x*image_width),int(face_landmarks.landmark[INDEX].y*image_height)), 1,B_COLOR, 1) # dots areound R eyes
                x = int(face_landmarks.landmark[INDEX].x * image_width)
                y = int(face_landmarks.landmark[INDEX].y*image_height)
                Rrecords.append([x,y])
            for INDEX in INDEXES_LIST_LEFT:
                cv2.circle(frame,(int(face_landmarks.landmark[INDEX].x*image_width), int(face_landmarks.landmark[INDEX].y*image_height)), 1, B_COLOR, 1)
                x = int(face_landmarks.landmark[INDEX].x * image_width)
                y = int(face_landmarks.landmark[INDEX].y*image_height)
                Lrecords.append([x,y])
            D6 = Rrecords[6]
            D9 = Rrecords[9]
            L6 = Lrecords[6]
            L9 = Lrecords[9]
            distance1 = ((D9[0]-D6[0])**2 + (D9[1]-D6[1])**2)**1/6
            cv2.circle(frame, ((D6[0] + D9[0])//2, (D6[1] + D9[1])//2), int(distance1), INITIAL_COLOR, 1)
            cv2.circle(frame, ((L6[0] + L9[0]) // 2, (L6[1] + L9[1]) // 2), int(distance1), INITIAL_COLOR, 1)

            if eyes_areas.__len__() > 4:
                eyes_areas.pop(0)
            if distance1*100/np.mean(eyes_areas) < 20:
                INITIAL_COLOR = (0,255,0)
                B_COLOR = (0,255,0)
                NUMBER_OF_BLINKS += 1
            elif counter != 0:
                counter += 1
                if counter > 3:
                    counter = 0
                pass
            elif distance1*100/np.mean(eyes_areas) < 50:
                pass
            else:
                INITIAL_COLOR = (0,0,255)
                B_COLOR = (0,255,255)
                eyes_areas.append(distance1)
            cvzone.putTextRect(frame, f"Number of Blinks: {str(NUMBER_OF_BLINKS)}", (50,50), 3,1)
            #cv2.putText(frame, str(NUMBER_OF_BLINKS), (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
            plotimg = PLOTLY.update(distance1)
            final_frame = cvzone.stackImages([plotimg, frame], 2, 1)

    cv2.imshow("Frame", final_frame) # Flip the image horizontally for a selfie-view dispay
    key = cv2.waitKey(100)


