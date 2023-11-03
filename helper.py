from ultralytics import YOLO
import time
import streamlit as st
import cv2
from pytube import YouTube
import settings


def load_model(model_path):
    # start_time = time.time()
    model = YOLO(model_path)
    # end_time = time.time()
    # print(f"Model loading time: {end_time - start_time} seconds")
    return model


def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        # tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker
    return is_display_tracker


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None):
    # start_time = time.time()

    # Resize the image to a standard size
    # image = cv2.resize(image, (720, int(720*(9/16))))
    image = cv2.resize(image, (480, int(480*(9/16))))  # Reduce the frame size

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    # end_time = time.time()
    # print(f"Drawing bounding boxes time: {end_time - start_time} seconds")
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )

def play_stored_video(conf, model):
    # start_time = time.time()
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    # is_display_tracker, tracker = display_tracker_options()
    is_display_tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker
                                             # tracker,
                                             )
                else:
                    vid_cap.release()
                    # end_time = time.time()
                    # print(f"Video remaking time: {end_time - start_time} seconds")
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
