from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'

# SOURCES_LIST = [IMAGE, VIDEO, WEBCAM, RTSP, YOUTUBE]
SOURCES_LIST = [IMAGE, VIDEO]

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'A2C1.png'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'A2C1_detected.jpg'

# Videos config
VIDEO_DIR = ROOT / 'videos'
VIDEO_1_PATH = VIDEO_DIR / 'AI비전로봇_1.mp4'
VIDEO_2_PATH = VIDEO_DIR / '20231102_0525_video_clip1.mp4'
VIDEO_3_PATH = VIDEO_DIR / 'm_density_slow2_1.mp4'
VIDEO_4_PATH = VIDEO_DIR / 'Made with Clipchamp(s_l).mp4'
# VIDEO_5_PATH = VIDEO_DIR / 'm_density_fast.mp4'
# VIDEO_6_PATH = VIDEO_DIR / 'm_density_slow2.mp4'
# VIDEO_7_PATH = VIDEO_DIR / 'm_density_very_fast.mp4'
# VIDEO_8_PATH = VIDEO_DIR / 'm_density_very_slow.mp4'
VIDEOS_DICT = {
    'Validation': VIDEO_4_PATH,
    'Test1': VIDEO_3_PATH,
    'Test2': VIDEO_2_PATH,
    'Test3': VIDEO_1_PATH,
    # 'video_1': VIDEO_1_PATH,
    # 'video_2': VIDEO_2_PATH,
    # 'video_3': VIDEO_3_PATH,
    # 'video_4': VIDEO_4_PATH,
    # 'video_5': VIDEO_5_PATH,
    # 'video_6': VIDEO_6_PATH,
    # 'video_7': VIDEO_7_PATH,
    # 'video_8': VIDEO_8_PATH,
}
# ML Model config
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'best3.pt'
