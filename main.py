import cv2
import random
trained_face_data = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml")

webcam = cv2.VideoCapture(0)

while True:
    frame_read_t_or_f, frame = webcam.read()
    gray_scaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(gray_scaled_img)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (random.randint(0, 256),
                                                  random.randint(0, 256), random.randint(0, 256)), random.randint(3, 6))
    cv2.imshow('lmao', frame)
    key = cv2.waitKey(1)

    if key==81 or key==113:
        break
webcam.release()
print("Code compiled succefully")   
