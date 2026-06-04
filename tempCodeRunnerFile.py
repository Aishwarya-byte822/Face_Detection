import cv2
import mediapipe as mp
import math

face_cascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml'
)


mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)

webcam = cv2.VideoCapture(0)


LEFT_EYE = [33, 160, 158, 133, 153, 144]

blink_count = 0
eye_closed = False


def distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


while True:

    success, img = webcam.read()

    if not success:
        break

    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        1.5,
        4
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


