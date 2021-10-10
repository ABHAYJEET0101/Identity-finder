import cv2
import numpy as np
import face_recognition

video_capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")


while True:
    ret, frame = video_capture.read()
    rgb_small_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2)
    # frame = cv2.cvtColor(frame, 0)
    detections = face_cascade.detectMultiScale(rgb_small_frame)
    face_locations = face_recognition.face_locations(rgb_small_frame)

    for x, y, z, w in detections:
        x *= 4
        y *= 4
        z *= 4
        w *= 4

        cv2.rectangle(frame, (w, x), (y, z), (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
