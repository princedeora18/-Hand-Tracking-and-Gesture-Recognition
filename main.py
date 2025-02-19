import cv2
import pygame
import mediapipe as mp
import math
import sys
import random

# Pygame setup
pygame.init()
WIDTH, HEIGHT = 1280, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Magic Hand Physics Demo")

# MediaPipe setup for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Camera setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not found!")
    sys.exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# Physics constants
GRAVITY = 0.5
FRICTION = 0.98
BOUNCE = 0.8
LAUNCH_FORCE = 15

# Colors
COLORS = {
    'background': (20, 20, 30),
    'skeleton': (200, 200, 200),
    'text': (255, 255, 0),
    'object': [(0, 255, 127), (255, 105, 180), (135, 206, 250)],
    'trail': (255, 255, 255, 50)
}


# Physics Object class
class PhysicsObject:
    def __init__(self):
        self.radius = random.randint(15, 25)
        self.pos = [random.randint(100, WIDTH - 100), random.randint(100, HEIGHT - 100)]
        self.vel = [0, 0]
        self.color = random.choice(COLORS['object'])
        self.trail = []
        self.type = random.choice(['normal', 'bouncy', 'sticky'])

    def update(self):
        # Apply physics
        self.vel[0] *= FRICTION
        self.vel[1] += GRAVITY
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]

        # Collision with walls
        if self.pos[0] < self.radius or self.pos[0] > WIDTH - self.radius:
            self.vel[0] *= -BOUNCE
        self.pos[0] = max(self.radius, min(WIDTH - self.radius, self.pos[0]))

        if self.pos[1] > HEIGHT - self.radius:
            self.pos[1] = HEIGHT - self.radius
            self.vel[1] *= -BOUNCE * (0.5 if self.type == 'sticky' else 1)

        # Keep a trail effect
        self.trail.append(self.pos.copy())
        if len(self.trail) > 15:
            self.trail.pop(0)


# Create physics objects
objects = [PhysicsObject() for _ in range(8)]
grabbed_object = None


def get_hand_landmarks():
    ret, frame = cap.read()
    if not ret:
        return None

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    cv2.imshow('Hand Tracking', cv2.resize(frame, (640, 360)))
    cv2.waitKey(1)

    if results.multi_hand_landmarks:
        return [(int(lm.x * WIDTH), int(lm.y * HEIGHT))
                for hand in results.multi_hand_landmarks
                for lm in hand.landmark]
    return None


def calculate_pinch(landmarks):
    """ Returns pinch distance and midpoint between thumb and index. """
    if len(landmarks) < 21:
        return 0, (0, 0)

    thumb = landmarks[4]
    index = landmarks[8]
    dist = math.hypot(index[0] - thumb[0], index[1] - thumb[1])
    mid = ((index[0] + thumb[0]) // 2, (index[1] + thumb[1]) // 2)

    return dist, mid


def main_loop():
    global grabbed_object
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)

    while True:
        screen.fill(COLORS['background'])
        landmarks = get_hand_landmarks()

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Hand tracking logic
        if landmarks:
            # Draw hand skeleton
            for connection in mp_hands.HAND_CONNECTIONS:
                pygame.draw.line(screen, COLORS['skeleton'], landmarks[connection[0]], landmarks[connection[1]], 2)

            # Pinch detection
            pinch_dist, pinch_pos = calculate_pinch(landmarks)
            pinch_strength = 1 - min(pinch_dist / 100, 1)

            # Grab/release objects
            if not grabbed_object and pinch_dist < 40:
                # Find closest object
                for obj in objects:
                    if math.hypot(obj.pos[0] - pinch_pos[0], obj.pos[1] - pinch_pos[1]) < obj.radius + 20:
                        grabbed_object = obj
                        obj.vel = [0, 0]
                        break
            elif grabbed_object and pinch_dist < 60:
                grabbed_object.pos = list(pinch_pos)
            else:
                if grabbed_object:
                    if grabbed_object.type == 'bouncy':
                        grabbed_object.vel[1] -= LAUNCH_FORCE * 2
                    grabbed_object = None

        # Update and draw objects
        for obj in objects:
            obj.update()

            # Draw trail
            for i, pos in enumerate(obj.trail):
                alpha = int(255 * (i / len(obj.trail)))
                pygame.draw.circle(screen, (COLORS['trail'][0], COLORS['trail'][1], COLORS['trail'][2], alpha),
                                   (int(pos[0]), int(pos[1])), int(obj.radius * (i / len(obj.trail))))

            # Draw object
            color = obj.color if obj != grabbed_object else tuple(min(c + 50, 255) for c in obj.color)
            pygame.draw.circle(screen, color, (int(obj.pos[0]), int(obj.pos[1])), obj.radius)

            # Draw type label
            text = font.render(obj.type[0].upper(), True, (255, 255, 255))
            screen.blit(text, (obj.pos[0] - 8, obj.pos[1] - 10))

        # Draw UI
        pygame.draw.rect(screen, (255, 255, 255), (20, HEIGHT - 60, 200, 20))
        pygame.draw.rect(screen, (0, 255, 127), (20, HEIGHT - 60, 200 * pinch_strength, 20))

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main_loop()
    cap.release()
    cv2.destroyAllWindows()
