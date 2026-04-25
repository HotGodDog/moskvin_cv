import cv2

cv2.namedWindow("Image", cv2.WINDOW_GUI_NORMAL)

image = cv2.imread("10. Contours/contours/defects.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(image, contours, 0, (255, 0, 0), 3)

rect = contours[0]
hull = cv2.convexHull(rect)
cv2.drawContours(image, [hull], -1, (255, 0, 255), 6)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows