import cv2
import pygame
import mediapipe as mp
import math
import sys

# Initialize Pygame and set up the window
pygame.init()
window_width = 1280
window_height = 720
screen = pygame.display.set_mode((window_width, window_height), pygame.DOUBLEBUF)
pygame.display.set_caption("Virtual Object Grab Demo")

# Set up MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    model_complexity=1
)

# Try to open a camera
cap = None
for i in range(3):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Found camera at index {i}")
        break
if not cap or not cap.isOpened():
    print("Error: No camera found! Using fallback pattern.")
    CAMERA_MODE = False
else:
    CAMERA_MODE = True
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Colors and settings
DEBUG_MODE = True
COLORS = {
    'background': (25, 25, 25),
    'skeleton': (200, 200, 200),
    'fingertips': (0, 255, 255),
    'text': (255, 255, 0),
    'warning': (255, 0, 0),
    'object': (0, 255, 0),
    'grabbed': (255, 0, 0)
}

# Virtual object parameters
obj_pos = [window_width // 2, window_height // 2]
obj_radius = 25
GRAB_THRESHOLD = 35
is_grabbed = False


def calculate_angle(p1, p2, p3):
    a = math.atan2(p3[1] - p2[1], p3[0] - p2[0]) - math.atan2(p1[1] - p2[1], p1[0] - p2[0])
    return abs(math.degrees(a))


def draw_debug_info(landmarks, gesture):
    pygame.draw.line(screen, COLORS['skeleton'], (50, 50), (150, 50), 2)
    pygame.draw.line(screen, COLORS['skeleton'], (50, 50), (50, 150), 2)
    if landmarks:
        xs = [p[0] for p in landmarks]
        ys = [p[1] for p in landmarks]
        bbox = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
        pygame.draw.rect(screen, COLORS['warning'], bbox, 2)


def main_loop():
    global obj_pos, is_grabbed
    clock = pygame.time.Clock()

    while True:
        screen.fill(COLORS['background'])
        index_tips = []

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if CAMERA_MODE:
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                if DEBUG_MODE:
                    cv2.imshow('Camera Debug', cv2.resize(frame, (640, 360)))
                    cv2.waitKey(1)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            x = int(lm.x * window_width)
                            y = int(lm.y * window_height)
                            landmarks.append((x, y))
                        index_tips.append(landmarks[8])

                        # Draw hand skeleton
                        for connection in mp_hands.HAND_CONNECTIONS:
                            start = landmarks[connection[0]]
                            end = landmarks[connection[1]]
                            pygame.draw.line(screen, COLORS['skeleton'], start, end, 3)

                        # Draw fingertips
                        for idx in [4, 8, 12, 16, 20]:
                            pygame.draw.circle(screen, COLORS['fingertips'], landmarks[idx], 10)

                # Object grabbing logic
                if results.multi_hand_landmarks:
                    if is_grabbed:
                        closest_distance = float('inf')
                        closest_tip = None
                        for tip in index_tips:
                            distance = math.hypot(tip[0] - obj_pos[0], tip[1] - obj_pos[1])
                            if distance < closest_distance:
                                closest_distance = distance
                                closest_tip = tip
                        if closest_distance <= GRAB_THRESHOLD:
                            obj_pos = list(closest_tip)
                        else:
                            is_grabbed = False
                    else:
                        for tip in index_tips:
                            distance = math.hypot(tip[0] - obj_pos[0], tip[1] - obj_pos[1])
                            if distance <= GRAB_THRESHOLD:
                                is_grabbed = True
                                obj_pos = list(tip)
                                break
                else:
                    is_grabbed = False
            else:
                font = pygame.font.Font(None, 74)
                text = font.render("Camera Error!", True, COLORS['warning'])
                screen.blit(text, (window_width // 3, window_height // 2))
        else:
            pygame.draw.circle(screen, COLORS['warning'],
                               (window_width // 2, window_height // 2), 50)

        # Draw virtual object
        color = COLORS['grabbed'] if is_grabbed else COLORS['object']
        pygame.draw.circle(screen, color, (int(obj_pos[0]), int(obj_pos[1])), obj_radius)

        if DEBUG_MODE:
            draw_debug_info(landmarks if 'landmarks' in locals() else None, None)

        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    main_loop()
    if CAMERA_MODE:
        cap.release()
    cv2.destroyAllWindows()