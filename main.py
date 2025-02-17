import cv2
import pygame
import mediapipe as mp
import math
import sys

# Initialize Pygame with double buffering
pygame.init()
window_width = 1280
window_height = 720
screen = pygame.display.set_mode((window_width, window_height), pygame.DOUBLEBUF)
pygame.display.set_caption("Advanced Hand Tracking with Debug")

# MediaPipe configuration with better parameters
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    model_complexity=1
)

# Camera initialization with multiple fallbacks
cap = None
for i in range(3):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Found camera at index {i}")
        break
if not cap or not cap.isOpened():
    print("Error: No camera found! Using test pattern.")
    CAMERA_MODE = False
else:
    CAMERA_MODE = True
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Debug visualization parameters
DEBUG_MODE = True  # Toggle this for debug info
COLORS = {
    'background': (25, 25, 25),
    'skeleton': (200, 200, 200),
    'fingertips': (0, 255, 255),
    'text': (255, 255, 0),
    'warning': (255, 0, 0)
}


def calculate_angle(p1, p2, p3):
    """Calculate angle between three points"""
    a = math.atan2(p3[1] - p2[1], p3[0] - p2[0]) - math.atan2(p1[1] - p2[1], p1[0] - p2[0])
    return abs(math.degrees(a))


def draw_debug_info(landmarks, gesture):
    """Draw debug information directly on Pygame surface"""
    # Draw coordinate system
    pygame.draw.line(screen, COLORS['skeleton'], (50, 50), (150, 50), 2)  # X-axis
    pygame.draw.line(screen, COLORS['skeleton'], (50, 50), (50, 150), 2)  # Y-axis

    # Draw hand bounding box
    if landmarks:
        xs = [p[0] for p in landmarks]
        ys = [p[1] for p in landmarks]
        bbox = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
        pygame.draw.rect(screen, COLORS['warning'], bbox, 2)


def main_loop():
    clock = pygame.time.Clock()
    last_gesture = None

    while True:
        screen.fill(COLORS['background'])

        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Camera processing
        if CAMERA_MODE:
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                if DEBUG_MODE:
                    # Show OpenCV preview in window
                    cv2.imshow('Camera Debug', cv2.resize(frame, (640, 360)))
                    cv2.waitKey(1)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            x = int(lm.x * window_width)
                            y = int(lm.y * window_height)
                            landmarks.append((x, y))

                        # Draw skeleton
                        for connection in mp_hands.HAND_CONNECTIONS:
                            start = landmarks[connection[0]]
                            end = landmarks[connection[1]]
                            pygame.draw.line(screen, COLORS['skeleton'], start, end, 3)

                        # Draw fingertips
                        for idx in [4, 8, 12, 16, 20]:
                            pygame.draw.circle(screen, COLORS['fingertips'],
                                               landmarks[idx], 10)

                        # Gesture recognition
                        # (Add your gesture recognition logic here)

                else:
                    # No hands detected message
                    font = pygame.font.Font(None, 74)
                    text = font.render("Show hands to camera!", True, COLORS['text'])
                    screen.blit(text, (window_width // 4, window_height // 2))
            else:
                # Camera error fallback
                font = pygame.font.Font(None, 74)
                text = font.render("Camera Error!", True, COLORS['warning'])
                screen.blit(text, (window_width // 3, window_height // 2))
        else:
            # No camera fallback pattern
            pygame.draw.circle(screen, COLORS['warning'],
                               (window_width // 2, window_height // 2), 50)

        if DEBUG_MODE:
            draw_debug_info(landmarks if 'landmarks' in locals() else None, last_gesture)

        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    main_loop()
    if CAMERA_MODE:
        cap.release()
    cv2.destroyAllWindows()