from ultralytics import YOLO
import streamlit as st
import cv2
import torch
import settings
from collections import Counter

real_names = {
    0 : '종이', #'Paper',
    1 : '종이팩', #'Paper_pack',
    2 : '종이컵', #'Paper_cup',
    3 : '비닐', #'Vinyl',
    4 : '비닐+이물질', #'Vinyl(f)',
    5 : '플라스틱', #'Plastic',
    6 : '플라스틱+이물질', #'Plastic(f)',
    7 : '재사용유리', #'Recycle_glass',
    8 : '갈색유리', #'Brown_glass',
    9 : '녹색유리', #'Green_glass',
    10 : '백색유리', #'White_glass',
    11 : '기타유리', #'Other_glass',
    12 : '캔류', #'Can',
    13 : '페트', #'Pet',  
    14 : '스티로폼', #'Form',
    15 : '건전지', #'Battery',
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

    col1, col2 = st.columns(2)
    with col1:
        with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
            video_bytes = video_file.read()
        if video_bytes:
            st.video(video_bytes, format='video/MP4')

    with col2:
        if st.sidebar.button('Detect Video Objects'):
            try:
                vid_cap = cv2.VideoCapture(str(settings.VIDEOS_DICT.get(source_vid)))
                st_frame = st.empty()
                # frame_skip = 2  # Skip every N frames for faster playback
                detected_objects = []
                

                while vid_cap.isOpened():
                    # for i in range(frame_skip):
                    #     vid_cap.read()
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
