import cv2

face_cascade = cv2.CascadeClassifier('haar_cascades/haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('haar_cascades/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haar_cascades/haarcascade_smile.xml')


def detect(gray, frame):

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
        extract_rectangle_gray = gray[y:y+h, x:x+w]
        extract_rectangle_color = frame[y:y+h, x:x+w]

        eyes = eyes_cascade.detectMultiScale(extract_rectangle_gray, 1.1, 22)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(extract_rectangle_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 3)

        smile = smile_cascade.detectMultiScale(extract_rectangle_gray, 1.7, 22)

        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(extract_rectangle_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 3)

    return frame


vdo = cv2.VideoCapture(0)

while True:
    ret, frame = vdo.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Detector', canvas)
    if cv2.waitKey(1) == ord('q'):
        break

vdo.release()
cv2.destroyAllWindows()