import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe pose class.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize Mediapipe drawing class.
mp_drawing = mp.solutions.drawing_utils

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a) # First point
    b = np.array(b) # Mid point
    c = np.array(c) # End point
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle 

# Initialize video capture
cap = cv2.VideoCapture(0)

# Variables to store push-up count and stages
counter = 0
stage = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with Mediapipe Pose
    results = pose.process(image)
    
    # Convert the frame back to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Extract landmarks
    try:
        landmarks = results.pose_landmarks.landmark
        
        # Get coordinates
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        
        # Calculate the angle
        angle = calculate_angle(shoulder, elbow, wrist)
        
        # Visualize angle
        cv2.putText(image, str(angle), 
                    tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                            )
        
        # Push-up counter logic
        if angle > 160:
            stage = "up"
        if angle < 30 and stage == 'up':
            stage = "down"
            counter += 1
            print(f'Push-up count: {counter}')
        
    except Exception as e:
        print(f'Error: {e}')
        pass
    
    # Render push-up count
    cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
    
    cv2.putText(image, 'Push-ups', (15,12), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(image, str(counter), 
                (10,60), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
    
    # Render detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                              )               
    
    cv2.imshow('Push-up Counter', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
