import cv2
import mediapipe as mp
import os
import time

save_directory = "captured_gestures"
os.makedirs(save_directory, exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

gesture_counter = 0
capture_interval = 2

while cap.isOpened():
    success, img = cap.read()
    
    if success:
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                
                motions = []
                
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    motions.append([id, cx, cy])

                thumb_is_up = motions[4][2] < motions[3][2]
                index_is_up = motions[8][2] < motions[7][2]
                middle_is_up = motions[12][2] < motions[11][2]
                ring_is_up = motions[16][2] < motions[15][2]
                pinky_is_up = motions[20][2] < motions[19][2]

                if not thumb_is_up and not index_is_up and not middle_is_up and not ring_is_up and not pinky_is_up:
                    gesture = "S"  # Fist (S)
                elif not thumb_is_up and not index_is_up and not middle_is_up and not ring_is_up and not pinky_is_up:
                    gesture = "O"  # Gesture for "O" (All fingers closed in a circle)
                elif not thumb_is_up and index_is_up and middle_is_up and not ring_is_up and not pinky_is_up:
                    gesture = "R"  # R shape (Index and Middle)
                elif thumb_is_up and not index_is_up and not middle_is_up and not ring_is_up and pinky_is_up:
                    gesture = "Y"  # Y shape (Thumb and Pinky)
                elif thumb_is_up and index_is_up and pinky_is_up and not middle_is_up and not ring_is_up:
                    gesture = "I Love You"  # Gesture for "I Love You"
                else:
                    gesture = ""

                cv2.putText(img, gesture, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # if gesture in ["S", "O", "R", "Y", "I Love You"]:
                #     gesture_counter += 1
                #     filename = os.path.join(save_directory, f"gesture_{gesture_counter}_{gesture}.png")
                #     cv2.imwrite(filename, img)
                #     print(f"Saved: {filename}")
                #     time.sleep(capture_interval)
        
        cv2.imshow("Capture", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

