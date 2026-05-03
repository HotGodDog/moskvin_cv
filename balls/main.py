import cv2
import numpy as np
import json
from pathlib import Path


save_path = Path(__file__).parent
config_path = save_path / "config.json"

cv2.namedWindow("Image", cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow("Mask", cv2.WINDOW_GUI_NORMAL)

position = [0, 0]
clicked = False

def on_click(even, x, y, flags, params):
    if even == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at {x}, {y}")
        global position
        global clicked
        position = [x, y]
        clicked = True

cv2.setMouseCallback("Image", on_click)
cam = cv2.VideoCapture(0+cv2.CAP_DSHOW)

lower = None
upper = None
balls = {}
colors = ["Blue", "Yellow", "Green"]
i = 0

col_low_upp = {}

if config_path.exists():
    with config_path.open("r") as f:
        js = json.load(f)
        for color in colors:
            if color in js:
                col_low_upp[color] = (np.array(js[color][0], dtype="u1"), np.array(js[color][1], dtype="u1"))

while cam.isOpened():
    ret, frame = cam.read()

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

    if clicked:
        clicked = False
        color = hsv[position[1], position[0]]
        lower = np.clip(color * 0.9, 0, 255).astype("u1")
        upper = np.clip(color * 1.1, 0, 255).astype("u1")
        upper[1] = 255
        upper[2] = 255

        balls[colors[i]] = (lower.tolist(), upper.tolist())
        i += 1

    for col in col_low_upp:
        lower = col_low_upp[col][0]
        upper = col_low_upp[col][1]

        if lower is not None:
            inr = cv2.inRange(hsv, lower, upper)
            mask = cv2.morphologyEx(inr, cv2.MORPH_CLOSE, np.ones((5, 5), dtype="u1"))

            cv2.imshow("Mask", mask)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                contours = max(contours, key=cv2.contourArea)
                (x, y), radius = cv2.minEnclosingCircle(contours)
                
                if radius > 10:
                    x = int(x)
                    y = int(y)
                    radius = int(radius)
                    cv2.circle(frame, (x, y), radius, (0, 255, 255), 4)
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    cv2.imshow("Image", frame)

cam.release()
cv2.destroyAllWindows()

print(balls)

with config_path.open("w") as f:
    json.dump(
        balls, f
    )
