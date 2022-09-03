import cv2
import mediapipe as mp
import time


pTime = 0
cap = cv2.VideoCapture(2)

mpFaceDetection = mp.solutions.face_detection
mp.Draw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)


while True:

    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)
    if results.detections:
        for id, detection in enumerate(results.detections):
            #     print(id, detection)
            #     print(detection.score)
            print(detection.location_data.relative_bounding_box)

    # FPS
    cTime = time.time()  # redo dont understand need explanation
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
