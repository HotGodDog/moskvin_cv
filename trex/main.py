import cv2
import numpy as np
import mss
import pyautogui
import time


def get_game_region():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screen_width = monitor['width']
        screen_height = monitor['height']
    
    width = screen_width // 2
    height = screen_height // 2
    x = screen_width - width
    y = 0
    w = width
    h = height
    
    return (x, y, w, h)

def detect_obstacle(img_bgr, shift):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # The width of the obstacle detection frame
    x1 = w // 3 + w // 19
    x2 = x1 + w // 20

    # Coordinates of the distant obstacle
    x11 = x1 + shift
    x21 = x2 + shift
    y11 = h // 2 + h // 7 + h // 15
    y21 = y11 + h // 7 - h // 15

    # Coordinates of a nearby obstacle
    x12 = x1 - w // 6 + shift
    x22 = x2 - w // 6 + shift
    y12 = h // 2 + h // 7
    y22 = y12 + h // 7

    # Coordinates of the GAME OVER frame
    x13 = w // 3 + w // 30
    x23 = x13 + w // 4
    y13 = h // 2 + h // 20
    y23 = h // 2 + h // 10 - h // 40

    roi1 = gray[y11:y21, x11:x21]
    roi2 = gray[y12:y22, x12:x22]
    roi3 = gray[y13:y23, x13:x23]
    
    _, binary1 = cv2.threshold(roi1, 100, 255, cv2.THRESH_BINARY_INV)
    dark_ratio1 = np.sum(binary1 == 255) / (roi1.shape[0] * roi1.shape[1])
    obstacle1 = dark_ratio1 > 0.02

    _, binary2 = cv2.threshold(roi2, 100, 255, cv2.THRESH_BINARY_INV)
    dark_ratio2 = np.sum(binary2 == 255) / (roi2.shape[0] * roi2.shape[1])
    obstacle2 = dark_ratio2 > 0.02

    _, binary3 = cv2.threshold(roi3, 100, 255, cv2.THRESH_BINARY_INV)
    dark_ratio3 = np.sum(binary3 == 255) / (roi3.shape[0] * roi3.shape[1])
    game_over = dark_ratio3 > 0.05
    
    debug_img = img_bgr.copy()

    # label1 = "DANGER" if obstacle1 else "CLEAR"
    color1 = (0, 0, 255) if obstacle1 else (0, 255, 0)

    # label2 = "DANGER" if obstacle2 else "CLEAR"
    color2 = (0, 0, 255) if obstacle2 else (0, 255, 0)

    label3 = "GAME OVER" if game_over else ""
    color3 = (0, 0, 255) if game_over else (0, 255, 0)

    # cv2.putText(debug_img, label1, (x11, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color1, 2)
    cv2.rectangle(debug_img, (x11, y11), (x21, y21), color1, 2)

    # cv2.putText(debug_img, label2, (x12, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color2, 2)
    cv2.rectangle(debug_img, (x12, y12), (x22, y22), color2, 2)

    cv2.putText(debug_img, label3, (x13, y13-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color3, 2)
    cv2.rectangle(debug_img, (x13, y13), (x23, y23), color3, 2)
        
    return obstacle1, obstacle2, game_over, debug_img

x, y, w, h = get_game_region()
capture_zone = {"left": x, "top": y, "width": w, "height": h}
last_obstacle_state = False

# Speed calculation data from the T-Rex game
INITIAL_SPEED = 6.0     
MAX_SPEED = 13.0        
ACCEL = 0.001          # Acceleration per frame
FPS = 60               # Approximate FPS for real-time translation
SECONDS_TO_MAX = (MAX_SPEED - INITIAL_SPEED) / (ACCEL * FPS) # ≈ 116 sec.

# The shift data is in my program
MAX_SHIFT = int(w / 6 + w / 20 + w / 40)     # Maximum shif at maximum speed

# The game starts 3 seconds after the program stars in order to have time to open the T-Rex window
time.sleep(3)
pyautogui.press('space')
game_start_time = time.time()

with mss.mss() as sct:
    while True:
        img = np.array(sct.grab(capture_zone))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        elapsed = time.time() - game_start_time     # The elapsed time since the beginning of the game

        # Calculating the current speed in the T-Rex game
        current_speed = INITIAL_SPEED + ACCEL * (elapsed * FPS)
        current_speed = min(current_speed, MAX_SPEED)

        # Calculation of the current shift
        speed_int = int(current_speed)

        idx = max(0, min(7, speed_int - 6))

        shift_coeffs = [0.00, 0.02, 0.04, 0.06, 0.08, 0.11, 0.14, 0.17]

        shift = int(w * shift_coeffs[idx])
        
        obstacle1, obstacle2, game_over, debug_img = detect_obstacle(img_bgr, shift)

        if last_obstacle_state and not obstacle1:
            pyautogui.press('space')

        if obstacle2:
            pyautogui.press('space')

        if game_over:
            time.sleep(3)
            pyautogui.press('space')
            game_start_time = time.time()

        last_obstacle_state = obstacle1

        # Выводим отладочную информацию на экран
        info_text = f"Speed: {current_speed:.2f}/{MAX_SPEED}, Shift: {shift}px"
        cv2.putText(debug_img, info_text, (w // 18, h // 2 + h // 3 + h // 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.imshow("Game Capture", debug_img)

        if cv2.waitKey(1) == ord('q'):
            break
        
        time.sleep(0.001)
 
cv2.destroyAllWindows()
