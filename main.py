from object_detector import *
from detect_holes import *
from kuka_cam_simulation import *
from shortest_path import *
from frame_transforms import *
from kuka_controller import *
import numpy as np


resize = True
resize_percentage = 0.5
resize_factor = 1
if resize:
    resize_factor = 1/resize_percentage

frame_height = 1200/resize_factor  #720
frame_width  = 1600/resize_factor  #1280

show_binarized_frame = True
show_contours = True  #show the simple approximation contours (inside "find_area_of_interest")
print_areas = True  #print the detected rectangles' areas
show_area_of_interest = True  #show the retrieved area of interest perimeter (containing the important holes)

# Algorithm's parameters that must be finetuned:
min_area_threshold = 40000  #150000  #the area of interest, containing the holes, should be the larger one
print_filter_holes_detection = False  #print information about the holes filtering
show_all_close_holes = True  #useful to tune holes filtering: show the frame with the "close" holes still in it
show_filtered_holes = True  #show the frame with the actual unique holes that have been detected

# Try to compensate for lens distortion (camera params are required)
undistort = True  # True=lens correction on
show_undistorted_frame = False

# Load Object Detector
detector = HomogeneousBgDetector()



# STEP 1) Read the camera frame, find the area of interest (base with holes)
raw_frame = get_camera_frame()

if undistort:  # Correct lens distortion
    raw_frame = undistort_frame(raw_frame, show_undistorted_frame)  #override the camera frame with the undistorted one
if resize:  # Resize raw frame if needed
    raw_frame = resize_frame(raw_frame, resize_percentage)

# Find the rectangular area of interest based on threshold and through binarization + OpenCV
area_of_interest, rect_corners = find_area_of_interest(raw_frame, min_area_threshold, show_binarized_frame, show_contours, show_area_of_interest, print_areas)



# STEP 2) Detect the holes inside the area of interest
detected_holes = False
detected_holes_all_frames = []
detected_radius_all_frames = []
frame_counter = 0
frame_with_holes = []
cv2.namedWindow("Holes detection"); cv2.moveWindow("Holes detection", 50, 30)

# detect and store all the holes' coordinates and radius inside N frames (now set to 50)
while not detected_holes:
    frame_counter += 1
    raw_frame = get_camera_frame()

    if undistort:  # Correct lens distortion
        raw_frame = undistort_frame(raw_frame, show_undistorted_frame=False)  # override the camera frame with the undistorted one
    if resize:  # Resize raw frame if needed
        raw_frame = resize_frame(raw_frame, resize_percentage)
    raw_frame_copy = raw_frame.copy()  # copy the raw frame (undistored if flag==True)

    frame_with_holes, holes_coords_single_frame, holes_radius_single_frame, num_detected_holes = detect_holes(raw_frame_copy)

    detected_holes_all_frames.append(holes_coords_single_frame)   #center coords x-y
    detected_radius_all_frames.append(holes_radius_single_frame)  #center radius

    # show the current frame with the detected holes
    cv2.imshow("Holes detection", frame_with_holes)
    cv2.waitKey(1)

    if frame_counter == 50:  #exit the loop when 50 frames have been collected
        break

# STEP 3) Filter the detected holes:
# Now holes coordinates are filtered in order to keep only the "more reliable ones"
# (by now they're filtered based on the euclidean distance, if necessary add a consensus policy)
threshold = 5  # euclidean distance threshold (TO BE TUNED)

# retrieve only "unique" centers
unique_centers, x1, x2, y1, y2 = filter_holes_detection(raw_frame.copy(), detected_holes_all_frames, detected_radius_all_frames, threshold, rect_corners, print_filter_holes_detection, show_all_close_holes, show_filtered_holes)
print("\nDetected holes [center_x, center_y, radius] (" + str(len(unique_centers)) + "):")
for c in unique_centers:
    print(c)


# STEP 4) Make a graph from the pixels' coordinates and find the best path for the robot to follow (optional)
use_path_minimization = True
if use_path_minimization:
    graph = create_graph_from_coords(unique_centers)
    source_vertex = 1
    best_permutation, best_costs, lowest_cost = travellingSalesmanProblem(graph, source_vertex)
    print("Best permutation found:", best_permutation, "with cost="+str(lowest_cost))
    print("All costs are:", best_costs, "(sum="+str(sum(best_costs))+")")
    plot_graph_over_image(raw_frame.copy(), graph, unique_centers, best_permutation, best_costs)
else:
    best_permutation, best_costs, lowest_cost = range(len(unique_centers)), None, None

cv2.waitKey(0)
cv2.destroyAllWindows()

#sort centers according to the best permutation found
sorted_unique_centers = [ unique_centers[k] for k in best_permutation ]


# STEP 5) Try to map 2D pixel coordinates into 3D points using camera parameters matrices
print("Conversion from 2D pixels to 3D points...")

# Intrinsic parameters matrix
K = np.matrix([[1372.38754, 0., 777.61866], [0., 1373.25601, 575.6542], [0., 0., 1.]])

# Camera projection matrix
P = np.matrix([[1327.75806, 0., 776.8763, 0.], [0., 1335.82239, 576.12011, 0.], [0., 0., 1., 0.]])

R_t = np.linalg.inv(K).dot(P)  # extrinsic parameters matrix
R = R_t[0:4, 0:3]              #   - rotation matrix
t = R_t[0:4, 3]                #   - translation vector

X0 = np.array([0, 0, 0]).transpose()  # camera origin

lambda_scale = np.array([-0.6])  # defines the distance from camera origin to 3D plane (intensity of the direction vector inv(K * R) * x)

raw_frame = get_camera_frame()
if undistort:  # Correct lens distortion
    raw_frame = undistort_frame(raw_frame, show_undistorted_frame=False)  # override the camera frame with the undistorted one
if resize:  # Resize raw frame if needed
    raw_frame = resize_frame(raw_frame, resize_percentage)
raw_frame_copy = raw_frame.copy()  # copy the raw frame (undistored if flag==True)
overlay = raw_frame.copy()

for c in sorted_unique_centers:
    x = c[0:2]
    x = np.insert(x, 2, 1)

    # Correct formula: X = X0 + lambda_scale * np.linalg.inv(K * R) * x
    # but X0 = [0 0 0]'

    X = lambda_scale.dot( np.linalg.inv( np.matmul(K, R) ).dot(x) )

    x_rep = np.dot(P, np.transpose(np.insert(X, 3, 1)))
    x_rep[0:2] = x_rep[0:2] / x_rep[2]
    x_rep = np.delete(x_rep, 2).tolist()[0]

    print("2D:", x, "to 3D", X, "repr.", x_rep)

    # draw the center and its reprojection from 3d to 2d
    cv2.circle(overlay, (int(c[0]), int(c[1])), int(c[2]), (0, 165, 255), -1)  #real center
    cv2.circle(overlay, (int(x_rep[0]), int(x_rep[1])), int(c[2]), (255, 0, 0), 1)  #reprojection
    opacity = 0.7
    font = cv2.FONT_HERSHEY_SIMPLEX
    org_1, org_2 = (int(c[0] * 1.05), int(c[1] * 1.05)), (int(c[0] * 1.05), int(c[1] * 0.95))

    fontScale, thickness = 0.4, 1
    cv2.putText(overlay, "[pixels]" + str(int(c[0])) + "," + str(int(c[1])), org_1, font, fontScale, (255, 0, 0), thickness, cv2.LINE_AA)
    #cv2.putText(overlay, str(int(x_rep[0])) + "," + str(int(x_rep[1])), org_2, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(overlay, "[cm]" + str(round(X[0,0]*100,2)) + "," + str(round(X[0,1]*100,2)), org_2, font, fontScale, (0, 0, 255), thickness, cv2.LINE_AA)

    cv2.addWeighted(overlay, opacity, raw_frame_copy, 1 - opacity, 0, raw_frame_copy)
cv2.imshow("Reprojection", raw_frame_copy)
cv2.waitKey(0)


# STEP 6) TO DO-CHECK if reliable: Control the robot in order to reach the center first hole
#         Maybe try to use the above 2D->3D projections, if correct; probably a second alignment process is required afterwards
#         ---- clarifications needed ----
for k, considered_center in enumerate(sorted_unique_centers):
    # Fine tuning of the hole detection: to run after the robot is over the detected hole but has to precisely align.
    # It could be used as a closed loop confirmation or alignment correction.
    print_cnt = 0

    print("[Control loop action] Considered hole ("+str(k+1)+"/"+str(len(sorted_unique_centers))+"):", considered_center)

    aligned = False
    while not aligned:
        raw_frame = get_camera_frame()
        if undistort:  # Correct lens distortion
            raw_frame = undistort_frame(raw_frame, show_undistorted_frame=False)  # override the camera frame with the undistorted one
        if resize:  # Resize raw frame if needed
            raw_frame = resize_frame(raw_frame, resize_percentage)
        raw_frame_copy = raw_frame.copy()  # copy the raw frame (undistored if flag==True)

        # draw the "considered_center"
        overlay = raw_frame.copy()
        cv2.circle(overlay, (int(considered_center[0]), int(considered_center[1])), int(considered_center[2]), (0, 165, 255), -1)
        opacity = 0.4
        cv2.addWeighted(overlay, opacity, raw_frame_copy, 1 - opacity, 0, raw_frame_copy)

        # draw a green circle in the middle of the frame
        radius_center_circle = int(frame_height/10)
        cv2.circle(overlay, (int(frame_width/2), int(frame_height/2)), radius_center_circle, (0, 255, 0), -1)
        opacity = 0.2
        cv2.addWeighted(overlay, opacity, raw_frame_copy, 1 - opacity, 0, raw_frame_copy)

        # draw a smaller circle in the middle of the frame
        center_threshold = radius_center_circle * 0.08
        cv2.circle(overlay, (int(frame_width/2), int(frame_height/2)), int(center_threshold), (0, 0, 0), 2)
        opacity = 0.8
        cv2.addWeighted(overlay, opacity, raw_frame_copy, 1 - opacity, 0, raw_frame_copy)

        cv2.imshow("Alignment to the considered hole ("+str(k+1)+"/"+str(len(unique_centers))+")", raw_frame_copy)

        # Call the control action
        show_prints = True
        x_ref_cm, y_ref_cm, x_error_pix, y_error_pix = position_controller(considered_center, radius_center_circle, center_threshold, frame_width, frame_height, show_prints, )

        #ROS move_kuka_callback(x_ref_cm, y_ref_cm) ???

        # Shift the area of interest based on the camera movement
        x_shift_pix, y_shift_pix = -x_error_pix, -y_error_pix
        area_of_interest_corners = move_area_of_interest(rect_corners, x_shift_pix, y_shift_pix)

        # detect and store all the holes' coordinates and radius inside N frames (now set to 50)
        frame_counter = 0
        while not detected_holes:
            frame_counter += 1
            raw_frame = get_camera_frame()

            if undistort:  # Correct lens distortion
                raw_frame = undistort_frame(raw_frame, show_undistorted_frame=False)  # override the camera frame with the undistorted one
            if resize:  # Resize raw frame if needed
                raw_frame = resize_frame(raw_frame, resize_percentage)
            raw_frame_copy = raw_frame.copy()  # copy the raw frame (undistored if flag==True)

            frame_with_holes, holes_coords_single_frame, holes_radius_single_frame, num_detected_holes = detect_holes(raw_frame_copy)

            detected_holes_all_frames.append(holes_coords_single_frame)  # center coords x-y
            detected_radius_all_frames.append(holes_radius_single_frame)  # center radius

            # show the current frame with the detected holes
            cv2.imshow("[Alignment fase] Holes detection", frame_with_holes)
            cv2.waitKey(1)

            if frame_counter == 50:  # exit the loop when 50 frames have been collected
                break

        # Now holes coordinates are filtered in order to keep only the "more reliable ones"
        threshold = 5  # euclidean distance threshold (TO BE TUNED)
        # retrieve only "unique" centers
        unique_centers, x1, x2, y1, y2 = filter_holes_detection(raw_frame.copy(), detected_holes_all_frames,
                                                                detected_radius_all_frames, threshold, area_of_interest_corners,
                                                                print_filter_holes_detection, show_all_close_holes,
                                                                show_filtered_holes)
        print("\n[Alignment fase] Detected holes [center_x, center_y, radius] (" + str(len(unique_centers)) + "):")
        for c in unique_centers:
            print(c)

        cv2.waitKey(1)


cv2.waitKey(0)
cv2.destroyAllWindows()


