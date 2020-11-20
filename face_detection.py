# Displays faces identified using haar cascades
# Can also replace the inside of line 11 VideoCapture() to identify objects in a video.
import cv2

# load the different cascade types
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

# startup the video
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# detect faces/eyes while the video runs
while video.isOpened():
    # get the current video frame
    _, frame = video.read()

    # detect faces using the grayscale of the frame
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(frame_gray, 1.5, 3)

    # for each face
    for (x, y, w, h) in faces:
        # draw a rectangle around the face
        x2, y2 = x + w, y + h
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 3)

        # detect each eye on the face and draw a rectangle for it
        eyes = eye_classifier.detectMultiScale(frame_gray[y:y2, x:x2])
        for (ex, ey, ew, eh) in eyes:
            ex2, ey2 = ex + ew, ey + eh
            cv2.rectangle(frame[y:y2, x:x2], (ex, ey), (ex2, ey2), (0, 0, 255), 3)

    # show output with detected face
    cv2.imshow('Video with Faces', frame)

    # check for quit key
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == ord(" ") or key == 27:
        break

# close video and windows
video.release()
cv2.destroyAllWindows()