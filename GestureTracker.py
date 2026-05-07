import cv2 as cv
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class handDetector():
    def __init__(self, maxHands=2, detectionCon=0.5):
        self.maxHands = maxHands
        self.detectionCon = detectionCon

        base_options = python.BaseOptions(model_asset_path="C:\Python Program\ALL PROJECTS\Project_copy\AirInk\hand_landmarker.task")

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=self.maxHands,
            min_hand_detection_confidence=self.detectionCon
        )

        self.detector = vision.HandLandmarker.create_from_options(options)

        self.tipIds = [4, 8, 12, 16, 20]
        self.lmList = []

    def findHands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        self.results = self.detector.detect(mp_image)

        if self.results.hand_landmarks:
            for handLms in self.results.hand_landmarks:
                h, w, _ = img.shape
                for id, lm in enumerate(handLms):
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    if draw:
                        cv.circle(img, (cx, cy), 5, (255, 0, 255), cv.FILLED)

        return img

    def findPosition(self, img, handNo=0):
        self.lmList = []

        if self.results.hand_landmarks:
            hand = self.results.hand_landmarks[handNo]
            h, w, _ = img.shape

            for id, lm in enumerate(hand):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])

        return self.lmList

    def fingersUp(self):
        fingers = []

        if len(self.lmList) == 0:
            return []

        # Thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers


def main():
    pTime = 0
    capture = cv.VideoCapture(0)
    detector = handDetector()

    while True:
        success, frame = capture.read()
        if not success:
            break

        frame = detector.findHands(frame)
        lmList = detector.findPosition(frame)

        if len(lmList) != 0:
            print(lmList[4])

        frame = cv.flip(frame, 1)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(frame, str(int(fps)), (10, 70),
                   cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv.imshow("Image", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()