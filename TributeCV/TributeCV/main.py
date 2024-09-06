import mediapipe as mp
import cv2
from mediapipe.python.solutions import hands

videoCapture = cv2.VideoCapture(0)
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
while videoCapture.isOpened():
    while True:
        ret, img = videoCapture.read()
        # result = mp_pose.Pose.process(image=img,self=)
        # results = mp_hands.Hands.process(image=img)
        with mp_pose.Pose(static_image_mode=True) as pose:
            result = pose.process(img)

        with mp_hands.Hands(static_image_mode=True) as hands:
            results = hands.process(img)

        # red_dot = mp_draw.DrawingSpec(color=(0,0,255), thickness=-1,circle_radius=1)
        mp_draw.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        if result.pose_landmarks:
            if result.pose_landmarks.landmark[20].y < result.pose_landmarks.landmark[6].y:
                img = cv2.putText(img, "Hand raised", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                              (255, 0, 0), 2)

        cv2.imshow("Pose only", img)
        # mp_draw.draw_landmarks(img, landmark_list=results.multi_hand_landmarks, landmark_drawing_spec=mp_hands.HAND_CONNECTIONS)
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS)
                if hand_landmark.landmark[8].y < hand_landmark.landmark[6].y and hand_landmark.landmark[12].y < \
                        hand_landmark.landmark[10].y and hand_landmark.landmark[16].y < hand_landmark.landmark[14].y and \
                        hand_landmark.landmark[20].y > hand_landmark.landmark[18].y and hand_landmark.landmark[4].x < \
                        hand_landmark.landmark[2].y:
                    if result.pose_landmarks.landmark[20].y < result.pose_landmarks.landmark[6].y:
                        img = cv2.putText(img, "We have a tribute", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                      (255, 0, 0), 2)

        cv2.imshow("hands", img)
        key = cv2.waitKey(1)
        if key == 27:
            break
cv2.destroyAllWindows()
