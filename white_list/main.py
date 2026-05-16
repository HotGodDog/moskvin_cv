import cv2
import numpy as np
import zmq


context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.setsockopt(zmq.SUBSCRIBE, b"")
socket.connect("tcp://84.237.21.36:6002")

cv2.namedWindow("Stream", cv2.WINDOW_GUI_NORMAL)

count = 0
while True:
    msg = socket.recv()
    # print(len(msg))

    key = cv2.waitKey(100)
    if key == ord('q'):
        break

    count += 1
    frame = cv2.imdecode(np.frombuffer(msg, np.uint8), -1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bin = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    bin = cv2.bitwise_not(bin)
    contours, _ = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    arrow = contours[0]
    eps = 0.01 * cv2.arcLength(arrow, True)
    approx = cv2.approxPolyDP(arrow, eps, True)
    for p in approx:
        cv2.circle(frame, tuple(*p), 6, (0, 255, 0), 2)

    rect = cv2.minAreaRect(arrow)
    bbox = cv2.boxPoints(rect)
    bbox = np.int32(bbox)
    print(bbox)
    cv2.drawContours(frame, [bbox], 0, (0, 255, 0), 2)

    text = np.zeros((360, 640, 3), dtype=np.uint8)
    info = f"I HATE T-Rex"
    cv2.putText(text, info, (text.shape[1] // 3, text.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    rows, cols, _ = text.shape

    pts1 = np.array([[0, 0], [cols, 0], [cols, rows], [0, rows]], dtype="f4")
    pts2 = np.array(bbox, dtype="f4")

    m = cv2.getPerspectiveTransform(pts1, pts2)

    trans = cv2.warpPerspective(text, m, (frame.shape[1], frame.shape[0]))

    gray = cv2.cvtColor(trans, cv2.COLOR_BGR2GRAY)

    ret, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
    fg = cv2.bitwise_and(trans, trans, mask=mask)

    result = cv2.add(bg, fg)

    cv2.putText(result, f"Count {count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0))
    cv2.imshow("Stream", result)
