import cv2
import mediapipe as mp
import numpy as np

# Video capture settings
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Mediapipe hand tracking setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)  # Process only one hand at a time for simplicity
mpDraw = mp.solutions.drawing_utils

# Function to detect gestures
def detect_gesture(motions):
    thumb_is_open = motions[4][1] < motions[3][1]  # Thumb check
    fingers_status = [motions[i][2] < motions[i-2][2] for i in [8, 12, 16, 20]]  # Status for each finger (open or closed)

    # Gesture recognition logic (simple example)
    if not thumb_is_open and all(not status for status in fingers_status):  # Check if all fingers are closed
        return "Fist"  # All fingers closed -> Fist gesture (ASL 'A')
    elif all(fingers_status):  # Check if all fingers are open
        return "Open Hand"  # All fingers open -> Open hand (ASL 'B' for example)
    return "Unknown"

while cap.isOpened():
    success, img = cap.read()
    
    if success:
        # Optional: Apply mirror effect
        img = cv2.flip(img, 1)
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process hand landmarks
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                # Draw landmarks on the hand
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                
                # List to hold the position of the hand landmarks
                motions = []
                
                # Extract landmark positions
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    motions.append([id, cx, cy])

                # Detect gesture
                if len(motions) > 20:  # Ensure all landmarks are detected
                    gesture = detect_gesture(motions)
                    
                    # Draw gesture on the image
                    cv2.putText(img, gesture, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Show the output frame
        cv2.imshow("Capture", img)
        
        # Break the loop with the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
