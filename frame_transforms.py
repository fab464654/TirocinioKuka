
from camera_calibration_data import *
import cv2

# Undistort the frame
def undistort_frame(raw_frame, show_undistorted_frame):
    und_raw_frame = cv2.undistort(raw_frame, cam, distCoeff)

    if show_undistorted_frame:
        side_by_side_frames = np.concatenate((raw_frame, und_raw_frame), axis=1)
        side_by_side_frames = cv2.resize(side_by_side_frames, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)
        cv2.imshow("distorted vs undistorted raw_frame", side_by_side_frames)
        cv2.waitKey(0)
    return und_raw_frame

# Resize the frame
def resize_frame(raw_frame, resize_percentage):
    resized_frame = cv2.resize(raw_frame, None, fx=resize_percentage, fy=resize_percentage, interpolation=cv2.INTER_CUBIC)
    return resized_frame