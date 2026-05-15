import cv2
import numpy as np
import mss
import pyautogui
import time


INITIAL_SPEED = 6.0
MAX_SPEED = 13.0
ACCEL = 0.001
FPS = 60

MAX_SHIFT = 250


def select_game_region():
    """Выделяем область игры мышью на всём виртуальном экране"""
    with mss.MSS() as sct:
        img = np.array(sct.grab(sct.monitors[2]))[:, :, :-1]
    
    clone = img.copy()
    roi = [None]
    drawing = [False]
    start_pt = [None]
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing[0] = True
            start_pt[0] = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and drawing[0]:
            tmp = clone.copy()
            cv2.rectangle(tmp, start_pt[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("Select Game Region", tmp)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing[0] = False
            x1, y1 = start_pt[0]
            x2, y2 = x, y

            roi[0] = (
                min(x1, x2),
                min(y1, y2),
                max(x1, x2),
                max(y1, y2)
            )
            tmp = clone.copy()

            cv2.rectangle(tmp, start_pt[0], (x, y), (0, 0, 255), 3)
            cv2.imshow("Select Game Region", tmp)
    
    cv2.namedWindow("Select Game Region")
    cv2.setMouseCallback("Select Game Region", mouse_callback)
    cv2.imshow("Select Game Region", img)
    
    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('e') and roi[0]:
            break

        elif key == ord('q'):
            cv2.destroyAllWindows()
            raise SystemExit("Cancelled by user")
    
    cv2.destroyAllWindows()
    
    return roi[0]


def find_dino(capture_zone):
    """Тыкаем на динозавра, выделяем, возвращаем его координаты"""
    selected = [None]
    
    with mss.MSS() as sct:
        img = np.array(sct.grab(capture_zone))[:, :, :-1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        
        def click_callback(event, x, y, flags, param):
            """При клике ищем фигуру в 50×50 окне вокруг курсора"""
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            
            h, w = binary.shape
            x1 = max(0, x - 40)
            y1 = max(0, y - 40)
            x2 = min(w, x + 40)
            y2 = min(h, y + 40)
            
            roi_bin = binary[y1:y2, x1:x2]
            
            if np.sum(roi_bin == 255) == 0:
                return
            
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(roi_bin, connectivity=8)
            
            cx, cy = (x2 - x1) // 2, (y2 - y1) // 2
            best_label = -1
            best_dist = float('inf')
            
            for i in range(1, num_labels):
                comp_mask = (labels == i)
                comp_y, comp_x = np.where(comp_mask)
                comp_cx = np.mean(comp_x)
                comp_cy = np.mean(comp_y)
                
                dist = (comp_cx - cx) ** 2 + (comp_cy - cy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_label = i
            
            if best_label == -1:
                return
            
            fx = stats[best_label, cv2.CC_STAT_LEFT]
            fy = stats[best_label, cv2.CC_STAT_TOP]
            fw = stats[best_label, cv2.CC_STAT_WIDTH]
            fh = stats[best_label, cv2.CC_STAT_HEIGHT]
            
            abs_x = x1 + fx
            abs_y = y1 + fy
            
            selected[0] = (abs_x, abs_y, fw, fh)
            
            tmp = img.copy()
            cv2.rectangle(tmp, (abs_x, abs_y), (abs_x + fw, abs_y + fh), (255, 255, 0), 1)
            cv2.imshow("Capture", tmp)
        
        cv2.namedWindow("Capture")
        cv2.setMouseCallback("Capture", click_callback)
        cv2.imshow("Capture", img)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('e') and selected[0] is not None:
                break
            elif key == ord('q'):
                cv2.destroyAllWindows()
                raise SystemExit("Cancelled")
    
    cv2.destroyAllWindows()

    return selected[0]


def detect_obstacle(img_bgr, dino, shift):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    dino_x, dino_y, dino_w, dino_h = dino
    dino_x += 23
    
    x11 = dino_x + dino_w + 140 + shift
    x21 = x11 + 30
    y11 = dino_y + 20
    y21 = y11 + dino_h - 20
    
    x12 = dino_x + dino_w + 10 + shift
    x22 = x12 + 30
    y12 = dino_y
    y22 = y12 + dino_h
    
    x13 = dino_x + dino_w + 140
    x23 = x13 + 192
    y13 = dino_y - 55
    y23 = y13 + 15 
    
    roi1 = gray[y11:y21, x11:x21]
    roi2 = gray[y12:y22, x12:x22]
    roi3 = gray[y13:y23, x13:x23]
    
    _, bin1 = cv2.threshold(roi1, 100, 255, cv2.THRESH_BINARY_INV)
    dark1 = np.sum(bin1 == 255) / (roi1.shape[0] * roi1.shape[1])
    obstacle1 = dark1 > 0.1 

    _, bin2 = cv2.threshold(roi2, 100, 255, cv2.THRESH_BINARY_INV)
    dark2 = np.sum(bin2 == 255) / (roi2.shape[0] * roi2.shape[1])
    obstacle2 = dark2 > 0.05 

    _, bin3 = cv2.threshold(roi3, 100, 255, cv2.THRESH_BINARY_INV)
    dark3 = np.sum(bin3 == 255) / (roi3.shape[0] * roi3.shape[1])
    game_over = dark3  > 0.05
    
    debug = img_bgr.copy()
    
    color1 = (0, 0, 255) if obstacle1 else (0, 255, 0)
    color2 = (0, 0, 255) if obstacle2 else (0, 255, 0)
    label3 = "GAME OVER" if game_over else ""
    color3 = (255, 255, 0) if game_over else (0, 255, 0)

    cv2.rectangle(debug, (x11, y11), (x21, y21), color1, 2)
    cv2.rectangle(debug, (x12, y12), (x22, y22), color2, 2)

    cv2.putText(debug, label3, (x13+41, y13-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color3, 2)
    cv2.rectangle(debug, (x13, y13), (x23, y23), color3, 1)
    
    cv2.rectangle(debug, (dino_x, dino_y), (dino_x + dino_w, dino_y + dino_h), (255, 255, 0), 1)
    
    return obstacle1, obstacle2, game_over, debug


region = select_game_region()
x, y, w, h = region
capture_zone = {"left": x, "top": y, "width": w - x, "height": h - y}
dino = find_dino(capture_zone)

time.sleep(3)
pyautogui.press('space')
game_start_time = time.time()
last_obstacle_state = False

with mss.MSS() as sct:
    while True:
        img = np.array(sct.grab(capture_zone))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        elapsed = time.time() - game_start_time
        current_speed = INITIAL_SPEED + ACCEL * (elapsed * FPS)
        current_speed = min(current_speed, MAX_SPEED)

        # progress = (current_speed - INITIAL_SPEED) / (MAX_SPEED - INITIAL_SPEED)    
        # shift = int(MAX_SHIFT * progress)

        speed_int = int(current_speed)
        idx = max(0, min(7, speed_int - 6))
        shift_coeffs = [0.0, 0.12, 0.28, 0.4, 0.59, 0.7, 0.84, 1.0]
        shift = int(MAX_SHIFT * shift_coeffs[idx])
        
        obstacle1, obstacle2, game_over, debug = detect_obstacle(img_bgr, dino, shift)

        cv2.imshow("Bot", debug)

        if last_obstacle_state and not obstacle1:
            pyautogui.press('space')
        
        if obstacle2:
            pyautogui.press('space')
        
        if game_over:
            time.sleep(3)
            pyautogui.press('space')
            game_start_time = time.time()
        
        last_obstacle_state = obstacle1

        info = f"Speed: {current_speed:.2f}/{MAX_SPEED}, Shift: {shift}"
        cv2.putText(debug, info, (10, debug.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("Bot", debug)
        
        if cv2.waitKey(1) == ord('q'): 
            break
        
        time.sleep(0.001)

cv2.destroyAllWindows()
