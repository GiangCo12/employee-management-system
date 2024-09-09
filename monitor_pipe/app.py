import streamlit as st
import cv2
import numpy as np
from pipe import identify_persons_in_video

# Configuration Constants
CAMERA_ID = 0
MODEL_FILE = 'yolov8n.pt'

# Main Streamlit app function
def main():
    st.title("Person Identification and Verification")

    # Start capturing video from the camera
    video_capture = cv2.VideoCapture(CAMERA_ID)

    if video_capture.isOpened():
        st.write("Camera is ready")
    else:
        st.write("Error: Unable to access the camera.")
        return

    # Button to start identification process
    if st.button("Identify Persons"):
        st.write("Identification process started...")

        # Call the function to identify persons in video
        identify_persons_in_video(CAMERA_ID, MODEL_FILE, visualize=True)

        st.write("Identification process completed.")

if __name__ == "__main__":
    main()
