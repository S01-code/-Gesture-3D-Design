"""
🎨 GESTURE 3D DESIGNER
Create colorful 3D designs, text, and screen projections using only hand gestures.
No keyboard or mouse required!

REQUIREMENTS (install before running):
    pip install opencv-python mediapipe numpy pygame PyOpenGL PyOpenGL_accelerate Pillow

GESTURE CONTROLS:
    ✋ Open Hand (5 fingers)     → Draw/Paint mode - move to paint
    ✌️  Two fingers (index+middle) → 3D Text mode - move to place text
    🤘 Rock sign (index+pinky)   → Shape mode - draw 3D shapes
    👆 One finger (index only)   → Erase mode
    👊 Fist                      → Clear canvas
    🤏 Pinch (thumb+index close) → Color picker - move up/down to change color
    🖐️  Palm facing camera        → Projection mode - creates screen projection effect
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import random
from collections import deque

# ─── Constants ────────────────────────────────────────────────────────────────
WINDOW_NAME = "🎨 Gesture 3D Designer"
CANVAS_W, CANVAS_H = 1280, 720
BRUSH_RADIUS = 18
TEXT_OPTIONS = ["HELLO", "ART", "WOW", "3D", "CREATE", "MAGIC", "WAVE", "✨"]

# Color palette (HSV hue values → converted to BGR)
PALETTE_HUES = [0, 15, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
PALETTE_COLORS_BGR = []
for h in PALETTE_HUES:
    hsv = np.uint8([[[h // 2, 255, 255]]])  # OpenCV hue is 0-179
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    PALETTE_COLORS_BGR.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))

# Extra special colors
PALETTE_COLORS_BGR += [(255, 255, 255), (200, 200, 200), (50, 50, 50)]

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# ─── Helper Functions ─────────────────────────────────────────────────────────

def lerp(a, b, t):
    return a + (b - a) * t


def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def fingers_up(hand_landmarks):
    """Return list of booleans: [thumb, index, middle, ring, pinky]"""
    lm = hand_landmarks.landmark
    tips = [4, 8, 12, 16, 20]
    base = [2, 5, 9, 13, 17]
    up = []
    # Thumb: compare x for left/right hand (simplified)
    up.append(lm[4].x < lm[3].x)
    for i in range(1, 5):
        up.append(lm[tips[i]].y < lm[base[i]].y)
    return up


def get_gesture(up):
    """Classify gesture from fingers_up list."""
    thumb, index, middle, ring, pinky = up
    count = sum(up)

    if count == 0:
        return "fist"
    if count == 5:
        return "open_hand"
    if index and not middle and not ring and not pinky:
        return "one_finger"
    if index and middle and not ring and not pinky:
        return "two_fingers"
    if index and not middle and not ring and pinky:
        return "rock"
    if thumb and index and not middle and not ring and not pinky:
        return "pinch_open"
    # Pinch detection handled separately
    return "other"


def draw_3d_text(canvas, text, x, y, color, scale=1.0, depth_layers=6):
    """Draw pseudo-3D extruded text."""
    font = cv2.FONT_HERSHEY_DUPLEX
    fs = 1.2 * scale
    thickness = max(1, int(2 * scale))

    # Shadow/depth layers
    for i in range(depth_layers, 0, -1):
        alpha = i / depth_layers
        dark = tuple(int(c * alpha * 0.4) for c in color)
        cv2.putText(canvas, text, (x + i * 2, y + i * 2), font, fs, dark, thickness + 1, cv2.LINE_AA)

    # Main text with gradient effect (multiple thin layers)
    for i in range(3):
        bright = tuple(min(255, int(c + (255 - c) * i * 0.3)) for c in color)
        cv2.putText(canvas, text, (x - i, y - i), font, fs, bright, max(1, thickness - i), cv2.LINE_AA)

    # Highlight
    cv2.putText(canvas, text, (x, y), font, fs, color, thickness, cv2.LINE_AA)


def draw_3d_sphere(canvas, cx, cy, radius, color):
    """Draw a shaded 3D-looking sphere."""
    # Gradient sphere using multiple circles
    for r in range(radius, 0, -2):
        t = 1 - (r / radius)
        # Brighten toward highlight
        bright_factor = lerp(0.3, 1.0, t)
        c = tuple(min(255, int(ch * bright_factor)) for ch in color)
        cv2.circle(canvas, (cx, cy), r, c, -1)

    # Specular highlight
    hx = cx - radius // 3
    hy = cy - radius // 3
    hr = max(2, radius // 5)
    cv2.circle(canvas, (hx, hy), hr, (255, 255, 255), -1)
    cv2.circle(canvas, (hx, hy), max(1, hr // 2), (255, 255, 255), -1)


def draw_3d_cube(canvas, cx, cy, size, color, angle=0):
    """Draw a pseudo-3D cube."""
    s = size
    off = int(s * 0.4)

    # 2D front face corners
    pts_front = np.array([
        [cx - s, cy - s],
        [cx + s, cy - s],
        [cx + s, cy + s],
        [cx - s, cy + s]
    ])

    # Offset back face
    pts_back = pts_front + np.array([off, -off])

    # Draw back face
    dark = tuple(max(0, c - 60) for c in color)
    cv2.fillPoly(canvas, [pts_back], dark)
    cv2.polylines(canvas, [pts_back], True, (255, 255, 255), 1)

    # Draw connecting edges
    for i in range(4):
        cv2.line(canvas, tuple(pts_front[i]), tuple(pts_back[i]),
                 tuple(min(255, c + 30) for c in color), 2)

    # Draw front face
    mid = tuple(min(255, c + 40) for c in color)
    cv2.fillPoly(canvas, [pts_front], mid)
    cv2.polylines(canvas, [pts_front], True, (255, 255, 255), 2)

    # Highlight top-left edge
    cv2.line(canvas, tuple(pts_front[0]), tuple(pts_front[1]), (255, 255, 255), 2)


def draw_projection_effect(canvas, cx, cy, color, frame_count):
    """Draw a holographic screen projection effect."""
    t = frame_count * 0.05
    # Radiating concentric ellipses
    for i in range(1, 8):
        r = int(30 + i * 25 + math.sin(t + i) * 10)
        alpha_c = max(0, 255 - i * 30)
        c = tuple(min(255, int(ch * alpha_c / 255)) for ch in color)
        cv2.ellipse(canvas, (cx, cy), (r, r // 2), 0, 0, 360, c, 1)

    # Scan lines
    scan_y = int(cy - 80 + ((frame_count * 3) % 160))
    cv2.line(canvas, (cx - 120, scan_y), (cx + 120, scan_y),
             tuple(min(255, c + 100) for c in color), 2)

    # Corner brackets
    bracket_size = 20
    corners = [
        (cx - 100, cy - 60), (cx + 100, cy - 60),
        (cx - 100, cy + 60), (cx + 100, cy + 60)
    ]
    dirs = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
    for (bx, by), (dx, dy) in zip(corners, dirs):
        cv2.line(canvas, (bx, by), (bx + dx * bracket_size, by), color, 2)
        cv2.line(canvas, (bx, by), (bx, by + dy * bracket_size), color, 2)

    # Center crosshair
    cv2.line(canvas, (cx - 10, cy), (cx + 10, cy), color, 1)
    cv2.line(canvas, (cx, cy - 10), (cx, cy + 10), color, 1)


def draw_ribbon_trail(canvas, trail, color):
    """Draw a 3D ribbon-like paint trail."""
    if len(trail) < 4:
        return
    pts = list(trail)
    for i in range(1, len(pts)):
        t = i / len(pts)
        # Width tapers
        w = max(1, int(BRUSH_RADIUS * t))
        # Simulate 3D by drawing dark shadow slightly offset
        shadow_c = tuple(max(0, c - 80) for c in color)
        cv2.line(canvas, (pts[i-1][0]+3, pts[i-1][1]+3),
                 (pts[i][0]+3, pts[i][1]+3), shadow_c, w + 2)
        # Main stroke
        bright_c = tuple(min(255, int(c + 40 * t)) for c in color)
        cv2.line(canvas, pts[i-1], pts[i], bright_c, w)
        # Highlight
        if w > 4:
            hl_c = tuple(min(255, c + 120) for c in color)
            cv2.line(canvas, (pts[i-1][0]-1, pts[i-1][1]-1),
                     (pts[i][0]-1, pts[i][1]-1), hl_c, max(1, w // 3))


# ─── Main Application ─────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CANVAS_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CANVAS_H)

    # White canvas
    canvas = np.ones((CANVAS_H, CANVAS_W, 3), dtype=np.uint8) * 255

    # State
    color_idx = 0
    current_color = PALETTE_COLORS_BGR[0]
    trail = deque(maxlen=40)
    last_pos = None
    frame_count = 0
    text_idx = 0
    text_place_cooldown = 0
    shape_idx = 0
    shape_place_cooldown = 0
    particles = []  # [(x, y, vx, vy, color, life)]
    gesture_label = ""
    pinch_start_y = None
    pinch_color_start = None

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6
    )

    print("\n🎨 GESTURE 3D DESIGNER STARTED!")
    print("=" * 50)
    print("Show your hand to the webcam and start creating!")
    print("Press 'Q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1
        h, w = frame.shape[:2]

        # Process hand
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        # Update particles
        new_particles = []
        for p in particles:
            px, py, vx, vy, pc, life = p
            if life > 0:
                cv2.circle(canvas, (int(px), int(py)), max(1, int(life / 8)), pc, -1)
                new_particles.append((px + vx, py + vy, vx, vy * 0.9 + 0.5, pc, life - 1))
        particles = new_particles

        gesture = "none"
        if result.multi_hand_landmarks:
            lm_data = result.multi_hand_landmarks[0]
            lm = lm_data.landmark

            # Finger tip position (index finger)
            ix = int(lm[8].x * w)
            iy = int(lm[8].y * h)
            # Palm center
            px_c = int(lm[9].x * w)
            py_c = int(lm[9].y * h)
            # Thumb tip
            tx = int(lm[4].x * w)
            ty = int(lm[4].y * h)

            up = fingers_up(lm_data)
            gesture = get_gesture(up)

            # Detect pinch (thumb-index close)
            pinch_dist = dist((tx, ty), (ix, iy))
            is_pinching = pinch_dist < 40

            # ── FIST: Clear canvas ──────────────────────────────────────
            if gesture == "fist":
                canvas[:] = 255
                trail.clear()
                particles.clear()
                gesture_label = "✊ CLEARED!"

            # ── OPEN HAND: 3D Paint ─────────────────────────────────────
            elif gesture == "open_hand":
                trail.append((ix, iy))
                draw_ribbon_trail(canvas, trail, current_color)
                # Sparkle particles
                if frame_count % 3 == 0:
                    for _ in range(3):
                        vx = random.uniform(-3, 3)
                        vy = random.uniform(-4, -1)
                        particles.append((ix, iy, vx, vy, current_color, 20))
                gesture_label = "✋ PAINTING"
                last_pos = (ix, iy)

            # ── ONE FINGER: Erase ───────────────────────────────────────
            elif gesture == "one_finger":
                cv2.circle(canvas, (ix, iy), BRUSH_RADIUS * 2, (255, 255, 255), -1)
                trail.clear()
                gesture_label = "👆 ERASING"

            # ── TWO FINGERS: 3D Text ────────────────────────────────────
            elif gesture == "two_fingers":
                text_place_cooldown = max(0, text_place_cooldown - 1)
                if text_place_cooldown == 0:
                    scale = 1.5 + (1 - lm[8].y) * 2  # scale by hand height
                    draw_3d_text(canvas, TEXT_OPTIONS[text_idx % len(TEXT_OPTIONS)],
                                 ix - 40, iy, current_color, scale)
                    text_idx += 1
                    text_place_cooldown = 15
                gesture_label = "✌️  3D TEXT"
                trail.clear()

            # ── ROCK SIGN: 3D Shapes ────────────────────────────────────
            elif gesture == "rock":
                shape_place_cooldown = max(0, shape_place_cooldown - 1)
                if shape_place_cooldown == 0:
                    sz = 30 + int((1 - lm[8].y) * 60)
                    if shape_idx % 2 == 0:
                        draw_3d_sphere(canvas, ix, iy, sz, current_color)
                    else:
                        draw_3d_cube(canvas, ix, iy, sz // 2, current_color)
                    shape_idx += 1
                    shape_place_cooldown = 20
                gesture_label = "🤘 3D SHAPE"
                trail.clear()

            # ── PINCH: Color selector ───────────────────────────────────
            elif is_pinching:
                if pinch_start_y is None:
                    pinch_start_y = py_c
                    pinch_color_start = color_idx
                else:
                    delta = (pinch_start_y - py_c) / h
                    color_idx = int(pinch_color_start + delta * len(PALETTE_COLORS_BGR)) % len(PALETTE_COLORS_BGR)
                    current_color = PALETTE_COLORS_BGR[color_idx]
                gesture_label = "🤏 COLOR SELECT"
                trail.clear()
            else:
                pinch_start_y = None

            # ── PALM (other with 4-5 fingers): Projection ───────────────
            if sum(up) >= 4 and not is_pinching and gesture != "open_hand":
                draw_projection_effect(canvas, px_c, py_c, current_color, frame_count)
                gesture_label = "🖐️  PROJECTION"
                trail.clear()

            # Clear trail if not painting
            if gesture not in ("open_hand",):
                if gesture != "open_hand":
                    trail.clear()

            # Draw hand landmarks on frame (not canvas)
            mp_drawing.draw_landmarks(frame, lm_data, mp_hands.HAND_CONNECTIONS)

        else:
            trail.clear()
            gesture_label = "Show your hand!"

        # ── Composite: blend canvas with webcam feed ─────────────────────
        # Canvas on left, webcam feed small in corner
        display = canvas.copy()

        # Webcam preview (bottom-right corner)
        cam_h, cam_w = 160, 240
        cam_small = cv2.resize(frame, (cam_w, cam_h))
        # Border
        cv2.rectangle(cam_small, (0, 0), (cam_w-1, cam_h-1), (80, 80, 80), 2)
        display[CANVAS_H - cam_h - 10:CANVAS_H - 10,
                CANVAS_W - cam_w - 10:CANVAS_W - 10] = cam_small

        # ── UI Overlay ────────────────────────────────────────────────────
        # Color swatch
        cv2.rectangle(display, (10, 10), (70, 70), current_color, -1)
        cv2.rectangle(display, (10, 10), (70, 70), (50, 50, 50), 2)
        cv2.putText(display, "COLOR", (12, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)

        # Gesture label
        cv2.rectangle(display, (0, CANVAS_H - 40), (400, CANVAS_H), (20, 20, 20), -1)
        cv2.putText(display, gesture_label, (10, CANVAS_H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Legend (top right)
        legend = [
            "GESTURES:",
            "✋ Open Hand = Paint",
            "✌  2 Fingers = 3D Text",
            "🤘 Rock Sign = 3D Shape",
            "👆 1 Finger  = Erase",
            "✊ Fist      = Clear",
            "🤏 Pinch+Move= Color",
        ]
        for i, line in enumerate(legend):
            y = 20 + i * 22
            col = (30, 30, 30) if i > 0 else (0, 100, 200)
            fw = 1 if i > 0 else 2
            cv2.putText(display, line, (CANVAS_W - 280, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, fw)

        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("\n👋 Thanks for creating! Goodbye!")
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
