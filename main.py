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
pygame.display.set_caption("Hand Tracking with Debug")

# Set up MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    model_complexity=1
)

# Try to open a camera, if none found, use a fallback
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

# Colors for different elements
DEBUG_MODE = True  # Set to False to disable debug info
COLORS = {
    'background': (25, 25, 25),
    'skeleton': (200, 200, 200),
    'fingertips': (0, 255, 255),
    'text': (255, 255, 0),
    'warning': (255, 0, 0)
}

# Function to calculate angle between three points
def calculate_angle(p1, p2, p3):
    a = math.atan2(p3[1] - p2[1], p3[0] - p2[0]) - math.atan2(p1[1] - p2[1], p1[0] - p2[0])
    return abs(math.degrees(a))

# Function to draw debug information on the screen
def draw_debug_info(landmarks, gesture):
    # Draw coordinate system (axes)
    pygame.draw.line(screen, COLORS['skeleton'], (50, 50), (150, 50), 2)  # X-axis
    pygame.draw.line(screen, COLORS['skeleton'], (50, 50), (50, 150), 2)  # Y-axis

    # If landmarks exist, draw a bounding box around the hand
    if landmarks:
        xs = [p[0] for p in landmarks]
        ys = [p[1] for p in landmarks]
        bbox = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
        pygame.draw.rect(screen, COLORS['warning'], bbox, 2)

# Main loop where everything happens
def main_loop():
    clock = pygame.time.Clock()
    last_gesture = None

    while True:
        screen.fill(COLORS['background'])

        # Handle Pygame events (like closing the window)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # If camera is available, process the frame
        if CAMERA_MODE:
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)  # Mirror the image for natural view
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                if DEBUG_MODE:
                    # Show the camera feed for debugging purposes
                    cv2.imshow('Camera Debug', cv2.resize(frame, (640, 360)))
                    cv2.waitKey(1)

                # If hand landmarks are found, draw them
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            x = int(lm.x * window_width)
                            y = int(lm.y * window_height)
                            landmarks.append((x, y))

                        # Draw the skeleton (lines connecting hand points)
                        for connection in mp_hands.HAND_CONNECTIONS:
                            start = landmarks[connection[0]]
                            end = landmarks[connection[1]]
                            pygame.draw.line(screen, COLORS['skeleton'], start, end, 3)

                        # Draw the fingertips as circles
                        for idx in [4, 8, 12, 16, 20]:
                            pygame.draw.circle(screen, COLORS['fingertips'],
                                               landmarks[idx], 10)

                else:
                    # Display message if no hands are detected
                    font = pygame.font.Font(None, 74)
                    text = font.render("Show hands to camera!", True, COLORS['text'])
                    screen.blit(text, (window_width // 4, window_height // 2))
            else:
                # Show error if camera is not working
                font = pygame.font.Font(None, 74)
                text = font.render("Camera Error!", True, COLORS['warning'])
                screen.blit(text, (window_width // 3, window_height // 2))
        else:
            # If no camera, show fallback pattern
            pygame.draw.circle(screen, COLORS['warning'],
                               (window_width // 2, window_height // 2), 50)

        # If debug mode is enabled, draw debug info
        if DEBUG_MODE:
            draw_debug_info(landmarks if 'landmarks' in locals() else None, last_gesture)

        # Update the screen and control the frame rate
        pygame.display.flip()
        clock.tick(30)

# Run the program
if __name__ == "__main__":
    main_loop()
    if CAMERA_MODE:
        cap.release()
    cv2.destroyAllWindows()
