import cv2

# This python file simulates the behaviour of the Kuka cam, "get_camera_frame" function returns a single frame of the
# input video (recorded at ICE lab)
def get_camera_frame():
    # Load camera/video information
    video_camera_source = "video4_cut.mp4"  # prerecorded video from Kuka cam or "https://192.168.1.58:8080/video"
    #video_camera_source = "video2.avi"  # prerecorded video from Kuka cam or "https://192.168.1.58:8080/video"

    #video_camera_source = "https://192.168.1.37:8080/video"

    cap = cv2.VideoCapture(video_camera_source)  # my ip camera on Android

    retval, raw_frame = cap.read()  # read the current frame

    return raw_frame
