import cv2
import numpy as np

cv2.namedWindow("Image", cv2.WINDOW_GUI_NORMAL)

image = cv2.imread("10. Contours/contours/arrow.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(image, contours, -1, (255, 0, 0), 3)

arrow = contours[0]
print(f"Area = {cv2.contourArea(arrow)}")
print(f"Perimeter = {cv2.arcLength(arrow, True)}")

moments = cv2.moments(arrow)
print(f"Moments = {moments}")
print(f"Area(moments) = {moments["m00"]}")

cx = int(moments["m10"] / moments["m00"])
cy = int(moments["m01"] / moments["m00"])

cv2.circle(image, (cx, cy), 5, (0, 255, 0), 4)

eps = 0.001 * cv2.arcLength(arrow, True)
approx = cv2.approxPolyDP(arrow, eps, True)
for p in approx:
    cv2.circle(image, tuple(*p), 6, (0, 255, 0), 2)

hull = cv2.convexHull(arrow)
for i in range(1, len(hull)):
    cv2.line(image, tuple(*hull[i-1]), tuple(*hull[i]), (0, 255, 0), 2)

cv2.line(image, tuple(*hull[-1]), tuple(*hull[0]), (0, 255, 0), 2)

x, y, w, h = cv2.boundingRect(arrow)
cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

rect = cv2.minAreaRect(arrow)
bbox = cv2.boxPoints(rect)
bbox = np.int32(bbox)
cv2.drawContours(image, [bbox], 0, (0, 255, 0), 2)

(x, y), radius = cv2.minEnclosingCircle(arrow)
center = int(x), int(y)
radius = int(radius)
cv2.circle(image, center, radius, (0, 255, 0), 4)

ellipse = cv2.fitEllipse(arrow)
cv2.ellipse(image, ellipse, (0, 255, 0), 4)

cv2.imshow("Image", image)
cv2.waitKey(0)

cv2.destroyAllWindows()