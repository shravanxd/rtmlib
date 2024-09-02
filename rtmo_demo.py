import time
import cv2
import numpy as np
from rtmlib import Body, draw_skeleton
import matplotlib.pyplot as plt
from collections import deque

# Initialize device and backend
device = 'cpu'
backend = 'onnxruntime'  # opencv, onnxruntime, openvino

# Initialize video capture and body detection model
cap = cv2.VideoCapture(0)
openpose_skeleton = False  # True for openpose-style, False for mmpose-style

body = Body(
    pose='rtmo',
    to_openpose=openpose_skeleton,
    mode='balanced',  # balanced, performance, lightweight
    backend=backend,
    device=device)

# Variables for tracking movement and graphing
frame_idx = 0
prev_keypoints = None
prev_velocity = None
movement_threshold = 0.2  # Adjust this threshold based on your data

# Initialize data storage for live graph
movement_data = deque(maxlen=100)  # Store last 100 frames of data
highlight_spikes = deque(maxlen=100)  # Store spikes for graph highlighting

# Set up live graph
plt.ion()  # Enable interactive mode
fig, ax = plt.subplots()
line, = ax.plot(movement_data, label='Movement Intensity')
highlight_line, = ax.plot(highlight_spikes, 'r', label='Violent Movement')
ax.set_ylim(0, 1)  # Normalized Y-axis (0 to 1)
ax.set_title("Real-Time Movement Intensity")
ax.set_xlabel("Frame")
ax.set_ylabel("Normalized Intensity")
plt.legend()

while cap.isOpened():
    success, frame = cap.read()
    frame_idx += 1

    if not success:
        break

    s = time.time()
    keypoints, scores = body(frame)
    det_time = time.time() - s
    print('det: ', det_time)

    img_show = frame.copy()

    # Calculate movement intensity (velocity)
    if prev_keypoints is not None:
        velocity = np.linalg.norm(keypoints - prev_keypoints, axis=-1)
        max_velocity = np.max(velocity)

        # Calculate acceleration to detect sudden changes
        if prev_velocity is not None:
            acceleration = np.abs(velocity - prev_velocity)
            max_acceleration = np.max(acceleration)
        else:
            max_acceleration = 0

        # Normalize the intensity
        normalized_intensity = max_velocity / (np.linalg.norm([frame.shape[0], frame.shape[1]]))
        normalized_acceleration = max_acceleration / (np.linalg.norm([frame.shape[0], frame.shape[1]]))

        print(f'Normalized Movement Intensity: {normalized_intensity}')
        print(f'Normalized Acceleration: {normalized_acceleration}')

        # Update live graph data
        movement_data.append(normalized_intensity)

        # Highlight violent movement based on combined velocity and acceleration
        if normalized_intensity > movement_threshold or normalized_acceleration > (movement_threshold / 2):
            highlight_spikes.append(normalized_intensity)
        else:
            highlight_spikes.append(0)

        line.set_ydata(movement_data)
        highlight_line.set_ydata(highlight_spikes)
        line.set_xdata(range(len(movement_data)))
        highlight_line.set_xdata(range(len(highlight_spikes)))

        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)  # Adjust the pause time as needed for smoother updating

        prev_velocity = velocity

    prev_keypoints = keypoints

    img_show = draw_skeleton(img_show,
                             keypoints,
                             scores,
                             openpose_skeleton=openpose_skeleton,
                             kpt_thr=0.3,
                             line_width=2)

    img_show = cv2.resize(img_show, (960, 640))
    cv2.imshow('img', img_show)
    cv2.waitKey(10)

# Close the video capture and matplotlib graph properly
cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
