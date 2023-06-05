from streamlit_webrtc import webrtc_streamer
import av
import cv2
import mediapipe as mp


mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    # print(result.multi_hand_landmarks)
    
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(img_rgb, handLms, mpHands.HAND_CONNECTIONS)
            
    return av.VideoFrame.from_ndarray(img_rgb, format="rgb24")


webrtc_streamer(key="gesture recognition", video_frame_callback=video_frame_callback)
