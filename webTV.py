import cv2
import time
import numpy as np

cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)
cam = cv2.VideoCapture(0+cv2.CAP_DSHOW)

tv = cv2.imread("9. OpenCV/news.jpg")

ret, frame = cam.read()
rows, cols, _ = frame.shape

pts1 = np.array([[0, 0], [cols, 0], [cols, rows], [0, rows]], dtype="f4")
pts2 = np.array([[18, 25], [432, 53], [435, 270], [39, 294]], dtype="f4")

tvsh1, tvsh0 = tv.shape[1], tv.shape[0]

prew_time = time.perf_counter()
while cam.isOpened():
    ret, frame = cam.read()

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    # curr_time = time.perf_counter()
    # print(f"FPS = {1 / (curr_time - prew_time):.1f}")
    # prew_time = curr_time

    m = cv2.getPerspectiveTransform(pts1, pts2)

    trans = cv2.warpPerspective(frame, m, (tvsh1, tvsh0))

    gray = cv2.cvtColor(trans, cv2.COLOR_BGR2GRAY)

    ret, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    bg = cv2.bitwise_and(tv, tv, mask=cv2.bitwise_not(mask))
    fg = cv2.bitwise_and(trans, trans, mask=mask)

    result = cv2.add(bg, fg)

    cv2.imshow("Camera", result)


cam.release()
cv2.destroyAllWindows()