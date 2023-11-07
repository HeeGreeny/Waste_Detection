from ultralytics import YOLO
import streamlit as st
import cv2
import torch
import settings
from collections import Counter

real_names = {
    0 : 'Paper',
    1 : 'Can',
    2 : 'Glass',
    3 : 'Pet',
    4 : 'Plastic',
    5 : 'Vinyl',
    6 : 'Styrofoam',
    7 : 'Battery',
    8 : 'Can(foreign)',
    9 : 'Glass(foreign)',
    10 : 'Pet(foreign)'
}

@st.cache_data
def load_model(model_path):
    model = YOLO(model_path)
    return model

def display_tracker_options():
    pass

def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None):
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))
    
    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)
    
    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted, caption='Detected Video',  channels="BGR",  use_column_width=True)      

def play_stored_video(conf, model):
    source_vid = st.sidebar.selectbox("Choose a video...", settings.VIDEOS_DICT.keys())
    is_display_tracker = display_tracker_options()

    def play_stored_video(conf, model):
    source_vid = st.sidebar.selectbox("Choose a video...", settings.VIDEOS_DICT.keys())
    is_display_tracker = display_tracker_options()
    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes, format='video/MP4')
    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            detected_objects = []
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                        _display_detected_frames(conf, model, st_frame, image, is_display_tracker)

                        # Calculate object counts after processing all frames
                        res = model.predict(image, conf=conf)
                        boxes = res[0].boxes
                        labels = boxes.cls
                        
                        for label in labels:
                                detected_objects.append(real_names[label.item()])                
                     
                else:
                        vid_cap.release()
                        break    

            if detected_objects:
                    detected_counts = Counter(detected_objects)
                    c_dict = dict(detected_counts) 

                # Create an expander to display the counts
                with st.expander("Detected Objects"):
                    for label, count in c_dict.items():
                        st.write(f"{label}")
                        
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
