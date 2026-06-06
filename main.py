import cv2
import mediapipe as mp

from utils import distance
from constants import *
face_cascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml'
)

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)

webcam = cv2.VideoCapture(0)

blink_count = 0
eye_closed = False

while True:
    success, img = webcam.read()

    if not success:
        break

    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.3,
        minNeighbors = 5
    )

    cv2.putText(
        img,
        f"Faces: {len(faces)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

  
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        for face in results.multi_face_landmarks:

            landmarks = face.landmark

            hor = distance(
                landmarks[LEFT_EYE[0]],
                landmarks[LEFT_EYE[3]]
            )

            ver = distance(
                landmarks[LEFT_EYE[1]],
                landmarks[LEFT_EYE[5]]
            )

            ratio = ver / hor

            cv2.putText(
                img,
                f"Ratio: {ratio:.2f}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

            if ratio < 0.20:

                if not eye_closed:
                    blink_count += 1
                    eye_closed = True

            else:
                eye_closed = False

            nose = landmarks[NOSE]
            left_face = landmarks[LEFT_FACE]
            right_face = landmarks[RIGHT_FACE]
            top_face = landmarks[TOP_FACE]  
            bottom_face = landmarks[BOTTOM_FACE]

            face_center_x = (
                left_face.x + right_face.x
            )/2

            face_center_y = (
                top_face.y + bottom_face.y
            )/2

            direction = "Center"

            if nose.x < face_center_x - 0.03:
                direction = "Looking Left"

            elif nose.x > face_center_x + 0.03:
                direction = "Looking Right"

            elif nose.y < face_center_y - 0.04:
                direction = "Lokking Up"

            elif nose.y > face_center_y + 0.04:
                direction = "Looking Down"


            cv2.putText(
                img,direction,(10,150),cv2.FONT_HERSHEY_SIMPLEX,1,
                (255,255,0),2
            )    

            h,w, _ = img.shape

            for point in [
                NOSE,LEFT_FACE,RIGHT_FACE,TOP_FACE,BOTTOM_FACE
            ]:  

                px = int(landmarks[point].x*w)
                py = int(landmarks[point].y*h)  

                cv2.circle(
                    img,
                    (px,py),4,(0,255,255),-1
                )  
    cv2.putText(
        img,
        f"Blinks: {blink_count}",
        (10, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.imshow("Face & Blink Detection", img)

    key = cv2.waitKey(1)

    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()
