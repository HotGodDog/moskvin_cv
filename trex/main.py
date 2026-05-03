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
    """
    Возвращает (True/False, изображение с нарисованной зоной, бинаризованное изображение зоны)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    x1 = w // 3 + w // 19
    x2 = x1 + w // 20
    y1 = h // 2 + h // 6 - h // 180
    y2 = y1 + h // 7

    x1 += shift
    x2 += shift

    roi = gray[y1:y2, x1:x2]
    
    # Бинаризация
    _, binary = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY_INV)
    dark_ratio = np.sum(binary == 255) / (roi.shape[0] * roi.shape[1])
    obstacle = dark_ratio > 0.02   # порог чувствительности
    
    
    # Копируем изображение для рисования
    debug_img = img_bgr.copy()
    label = "DANGER" if obstacle else "CLEAR"
    color = (0, 0, 255) if obstacle else (0, 255, 0)

    cv2.putText(debug_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
        
    return obstacle, debug_img

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
MAX_SHIFT = int(w / 6 + w / 20 + w / 320) + 12 # Maximum shif at maximum speed

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
        progress = (current_speed - INITIAL_SPEED) / (MAX_SPEED - INITIAL_SPEED)    
        shift = int(MAX_SHIFT * progress)
        
        obstacle, debug_img = detect_obstacle(img_bgr, shift)
        
        if last_obstacle_state and not obstacle:    
            pyautogui.press('space')

        last_obstacle_state = obstacle

        # Выводим отладочную информацию на экран
        info_text = f"Speed: {current_speed:.2f}/{MAX_SPEED}, Shift: {shift}px"
        cv2.putText(debug_img, info_text, (w // 18, h // 2 + h // 3 + h // 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.imshow("Game Capture", debug_img)

        if cv2.waitKey(1) == ord('q'):
            break
        
        time.sleep(0.001)
 
cv2.destroyAllWindows()
