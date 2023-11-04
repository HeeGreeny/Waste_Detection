# Python In-built packages
from pathlib import Path
import PIL
# import threading
# External packages
import streamlit as st
# Local Modules
import settings
import helper

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

# Set page layout
st.set_page_config(
    page_title="MS AI SCHOOL 1팀",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Main page heading
st.title("AI 생활 폐기물 분류 시스템")

# Sidebar
st.sidebar.header(" Upload trash images or Videos :fire:")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection'])
confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
    
# Load Pre-trained ML Model (Using st.cache to improve speed)
@st.cache_data
def load_model(model_path):
    model = helper.load_model(model_path)
    return model
try:
    model = load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header(":smile: Image / Video")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image, conf=confidence)
                boxes = res[0].boxes
                labels = boxes.cls
                bbox_coordinates = boxes.xyxy
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image', use_column_width=True)
                
                try:
                    col1, col2 = st.columns(2)
                    with col1:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.data)
                    with col2:
                        with st.expander("Detected Objects"):
                            # match real names to detected classes
                            # print out the objects detected with the total count of each name
                            
                            if len(labels) > 0:
                                labels = [real_names[float(l)] for l in labels]
                                # 각 클래스별로 몇개가 검출되었는지 출력
                                c = Counter(labels)
                                for label, count in c.items():
                                    st.write(f"{label} : {count}개")
                            else:
                                st.write("No objects detected!")
                 except Exception as ex:
                    st.write("No image is uploaded yet!")          

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

else:
    st.error("Please select a valid source type!")
