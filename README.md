# Gesture Recognition and Robotic Hand Synchronization

## Project Overview
This project aims to create a robotic hand that mimics human hand gestures detected via a webcam. Using MediaPipe for gesture recognition, the system calculates finger bending degrees based on joint positions and sends commands to servo motors via Arduino using serial communication. 

### Core Features
1. **Gesture Detection:**
   - Recognizes hand gestures through a webcam using MediaPipe.
   - Calculates finger bending degrees based on the distance between the fingertip and its corresponding base joint.

2. **Standardization:**
   - Ensures consistent detection regardless of hand size by overlaying a standardized hand shape for calibration.
   - Implements an `overlay_image` function to blend the calibration UI onto the webcam feed using OpenCV.

3. **Bending Degree to Servo Angle Conversion:**
   - Maps the distance between finger joints to servo motor angles using linear equations derived from experimental data.
   - Conversion functions are implemented for each finger.

4. **Serial Communication:**
   - Communicates between Python and Arduino using a fixed-length protocol to ensure error-free data transmission.
   - Implements `find_arduino()` to dynamically identify and connect to the correct Arduino port.

---

## Implementation Details

### Gesture Detection and Calibration
- Utilizes MediaPipe to calculate distances from each fingertip to the hand base.
- A standardized UI prompts users to align their hands to a predefined size for accurate bending degree calculations.
- Code: `find_distance.py` (distance calculation), `with_image.py` (overlay UI).

### Bending Degree Calculation
- Experimental measurements were taken for fully extended and fully bent positions:

| Finger | Distance (Extended) | Distance (Bent) | Slope | Intercept |
|--------|----------------------|-----------------|-------|-----------|
| Thumb  | 0.43                | 0.31            | 1500  | -465      |
| Index  | 0.74                | 0.26            | -375  | 277.5     |
| Middle | 0.80                | 0.24            | -272.7| 218.2     |
| Ring   | 0.76                | 0.22            | -333.3| 253.3     |
| Pinky  | 0.61                | 0.24            | -486.5| 296.8     |

- These values were used to create linear equations for each finger.
- Code: `find_angle.py`.

### Serial Communication Protocol
- Converts servo angles to a fixed 3-digit format using `".join(f"{math.floor(angle):03d}" for angle in angles)`.
- Ensures consistent slicing during Arduino processing.
- Utilizes `Serial.readString()` on Arduino to handle complete data packets.
- Code: `gesture_HCI.ino`.

---

## How to Use

### Prerequisites
- Python 3.x
- Arduino IDE
- Libraries: MediaPipe, OpenCV, pyserial

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/gesture_robot_hand.git
   cd gesture_robot_hand
   ```
2. Install required Python libraries:
   ```bash
   pip install mediapipe opencv-python pyserial
   ```
3. Run the calibration script to align hand size:
   ```bash
   python with_image.py
   ```
4. Upload the Arduino sketch to your Arduino board:
   ```bash
   # Use the Arduino IDE to upload `gesture_HCI.ino`
   ```
5. Start the main gesture-to-robot-hand program:
   ```bash
   python main.py
   ```

---

## Repository Structure
- `find_distance.py`: Calculates distances between finger joints.
- `with_image.py`: Displays the calibration UI.
- `find_angle.py`: Converts joint distances to servo angles.
- `gesture_HCI.ino`: Arduino code for receiving and processing servo commands.
- `main.py`: Integrates gesture recognition and robotic hand control.

---

## Future Work
- Optimize conversion functions using curve fitting and gradient descent.
- Enhance the calibration process for better usability.
- Integrate additional gestures beyond finger bending.

---

## Contributors
- [Your Name](https://github.com/your-github)

