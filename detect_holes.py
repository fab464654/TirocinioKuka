import cv2
import numpy as np
import random
from scipy.spatial.distance import cdist

# Initialize parameters of the SimpleBlobDetector
params = cv2.SimpleBlobDetector_Params()

# Filter by Area
params.filterByArea = True
params.minArea = 20  # 20000
params.maxArea = 3000  # 40000

# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.3

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.01

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.6

# Distance Between Blobs
params.minDistBetweenBlobs = 5

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

def detect_holes(frame):
    overlay = frame.copy()

    keypoints = detector.detect(frame)
    num_detected_holes = len(keypoints)

    for k in keypoints:
        cv2.circle(overlay, (int(k.pt[0]), int(k.pt[1])), int(k.size / 2), (0, 0, 255), -1)
        cv2.line(overlay, (int(k.pt[0]) - 20, int(k.pt[1])), (int(k.pt[0]) + 20, int(k.pt[1])), (0, 0, 0), 3)
        cv2.line(overlay, (int(k.pt[0]), int(k.pt[1]) - 20), (int(k.pt[0]), int(k.pt[1]) + 20), (0, 0, 0), 3)

    opacity = 0.5
    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

    holes_coords = [[k.pt[0], k.pt[1]] for k in keypoints]  #list of lists containing x and y coordinates of each hole's center
    holes_radius = [[k.size / 2] for k in keypoints]

    return frame, holes_coords, holes_radius, num_detected_holes


def unique(list1):
    # initialize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)

    return unique_list




def filter_holes_detection(frame, detected_holes_all_frames, detected_radius_all_frames, threshold, rect_corners, show_prints, show_all_close_holes, show_filtered_holes):
    #"detected_holes_all_frames" = list of all the detected holes (after dimensionality reduction)
    #       ----------- frame 1 ----------            ------------ frame 2 ------------
    # [ [hole1_x, hole1_y], [hole2_x, hole2_y] , [hole1_x, hole1_y], [hole2_x, hole2_y] ]
    #

    detected_center_coords = [holes_coords for detected_holes_single_frame in detected_holes_all_frames for holes_coords in detected_holes_single_frame]
    detected_center_radius = [holes_radius for detected_radius_single_frame in detected_radius_all_frames for holes_radius in detected_radius_single_frame]

    (topLeft, topRight, bottomLeft, bottomRight) = rect_corners

    x_min = min(topLeft[0], topRight[0], bottomRight[0], bottomLeft[0])
    x_max = max(topLeft[0], topRight[0], bottomRight[0], bottomLeft[0])
    y_min = min(topLeft[1], topRight[1], bottomRight[1], bottomLeft[1])
    y_max = max(topLeft[1], topRight[1], bottomRight[1], bottomLeft[1])

    if show_prints:
        print("[BEFORE FILTERING] Initial number of centers: ", len(detected_center_coords))
        print("[BEFORE FILTERING] Limits x_min, x_max, y_min, y_max:", x_min, x_max, y_min, y_max)

    actual_centers = []
    for k, center in enumerate(detected_center_coords):
        if (x_min < center[0] < x_max) and (y_min < center[1] < y_max):  #keep only if inside the x and y limits
            actual_centers.append([center[0], center[1], detected_center_radius[k][0]])

    if show_prints:
        print("[FILTERING STEP 1] Leftover number of centers: ", len(actual_centers))
        print("Detected actual_centers:")
        for c in actual_centers:
            print(c)

    if show_all_close_holes:
        overlay = frame.copy()
        for center in actual_centers:
            # To visualize and maybe distinguish threaded/not threaded holes (??)
            #color = np.random.randint(0,255)
            #color2 = np.random.randint(0,255)
            #cv2.circle(overlay, (int(center[0]), int(center[1])), int(center[2]), (color, 0, color2), 1)
            cv2.circle(overlay, (int(center[0]), int(center[1])), int(center[2]), (0, 0, 255), -1)
            cv2.line(overlay, (int(center[0]) - 20, int(center[1])), (int(center[0]) + 20, int(center[1])), (0, 0, 0), 3)
            cv2.line(overlay, (int(center[0]), int(center[1]) - 20), (int(center[0]), int(center[1]) + 20), (0, 0, 0), 3)

            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (int(center[0]*1.05), int(center[1]*1.05))
            fontScale, thickness = 0.4, 1
            color = (255, 0, 0)
            cv2.putText(overlay, str(int(center[0]))+"; "+str(int(center[1])), org, font, fontScale, color, thickness, cv2.LINE_AA)

        opacity = 0.5
        cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
        cv2.imshow("[FILTERING STEP 1] Filtering all the holes outside the region of interest", frame)


    # Filter for same center taken multiple times in same/different frames:
    # all the coordinates that refer to the same hole's center are collected and then averaged
    actual_centers_unique = [[0, 0, 0]]

    for center in actual_centers:
        #print("actual_centers len: ", len(actual_centers))
        actual_centers_copy_arr = np.array([np.array(center) for center in actual_centers])

        A = actual_centers_copy_arr
        B = np.array(center[:2])

        center_coords_to_average = A[(cdist(A[:, :2], B[None]) < threshold).ravel()]

        mean_candidate = np.mean(center_coords_to_average, axis=0)
        A_2 = np.array([np.array(center) for center in actual_centers_unique])
        B_2 = mean_candidate[:2]

        # To check:
        #print(A_2[(cdist(A_2[:, :2], B_2[None]) < threshold).ravel()] )
        #print(len(A_2[(cdist(A_2[:, :2], B_2[None]) < threshold).ravel()] ))

        if len(A_2[(cdist(A_2[:, :2], B_2[None]) < threshold).ravel()]) == 0:
            actual_centers_unique.append(mean_candidate)
        #else:  #print to check if holes are very close
            #print(mean_candidate, "too close to", A_2[(cdist(A_2[:, :2], B_2[None]) < threshold).ravel()] )

        for el in actual_centers:
            if el == center or el in center_coords_to_average:
                actual_centers.remove(el)

    actual_centers_unique.remove([0, 0, 0])
    if show_prints:
        print("[FILTERING STEP 2] number of centers: ", len(actual_centers_unique))

    if show_filtered_holes:
        frame_height, frame_width, _ = frame.shape
        overlay = frame.copy()
        filtered_frame = frame.copy()
        for center in actual_centers_unique:
            # Draw circle around the holes
            cv2.circle(overlay, (int(center[0]), int(center[1])), 10, (0, 255, 0), -1)
            # Draw diameters of the holes
            cv2.line(overlay, (int(center[0]) - 20, int(center[1])), (int(center[0]) + 20, int(center[1])), (0, 100, 0), 2)
            cv2.line(overlay, (int(center[0]), int(center[1]) - 20), (int(center[0]), int(center[1]) + 20), (0, 100, 0), 2)
            # Draw board limits
            cv2.line(overlay, (int(x_min), int(0)), (int(x_min), int(frame_height)), (255, 0, 0), 2)
            cv2.line(overlay, (int(x_max), int(0)), (int(x_max), int(frame_height)), (255, 0, 255), 2)
            cv2.line(overlay, (int(0), int(y_min)), (int(frame_width), int(y_min)), (255, 255, 0), 2)
            cv2.line(overlay, (int(0), int(y_max)), (int(frame_width), int(y_max)), (0, 255, 255), 2)
            cv2.rectangle(overlay, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)

        opacity = 0.5
        cv2.addWeighted(overlay, opacity, filtered_frame, 1 - opacity, 0, filtered_frame)
        # show the filtered frame
        cv2.namedWindow("[FILTERING STEP 2] Filtered holes detection")
        cv2.moveWindow("[FILTERING STEP 2] Filtered holes detection", 50, 30)
        cv2.imshow("[FILTERING STEP 2] Filtered holes detection", filtered_frame)
        cv2.waitKey(0)

    return actual_centers_unique, x_min, x_max, y_min, y_max



# Function to retrieve the rectangular area of interest
# source: https://docs.opencv.org/3.4/da/d0c/tutorial_bounding_rects_circles.html
def find_area_of_interest(raw_frame, min_area_threshold, show_binarized_frame, show_contours, show_area_of_interest, print_areas=False):

    # 1) Binarize the image: threshold values TO BE TUNED according to the scene
    frame_bw = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
    # ret, thresh1 = cv2.threshold(frame_bw, 160, 255, cv2.THRESH_BINARY)  #values for the ip android cam
    ret, thresh1 = cv2.threshold(frame_bw, 200, 255, cv2.THRESH_BINARY)  # values tuned for video4.avi
    # show the binarized image for tuning the above min and max thresholds
    if show_binarized_frame:
        cv2.imshow("Binarized frame", thresh1)
        cv2.waitKey(0)


    # 2) Extract contours of rectangles and draw them
    raw_frame_copy = raw_frame.copy()  #make a copy
    rect_corners, hierarchy2 = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(raw_frame_copy, rect_corners, -1, (0, 255, 0), 2, cv2.LINE_AA)
    if show_contours:
        cv2.imshow('SIMPLE Approximation contours', raw_frame_copy)


    # 3) Approximate contours to rectangles, compute the area and threshold it in order to find the area of interest
    #If all contours are needed (if not, only rectangles are considered)
    #contours, _ = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = rect_corners  #in my case it's sufficient to use rectangles
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 2, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

    # Draw polygonal contour + bonding rects + circles
    outer_rectangle = []
    if print_areas:
        print("Areas of the detected rectangles >"+ str(min_area_threshold/100) +" (threshold="+ str(min_area_threshold) +"):")

    for i in range(len(contours)):
        area = cv2.contourArea(contours_poly[i])
        if print_areas and area > min_area_threshold/100:
            print(area)
        if area > min_area_threshold:  #to filter by area
            color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
            cv2.drawContours(raw_frame, contours_poly, i, color)
            cv2.rectangle(raw_frame, (int(boundRect[i][0]), int(boundRect[i][1])), (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 5)
            outer_rectangle = boundRect[i] #save the outer perimeter rectangle

    topLeft = (int(outer_rectangle[0]), int(outer_rectangle[1] + outer_rectangle[3]))
    bottomLeft = (int(outer_rectangle[0]), int(outer_rectangle[1]))
    topRight = ((int(outer_rectangle[0]+ outer_rectangle[2])), int(outer_rectangle[1] + outer_rectangle[3]))
    bottomRight = ((int(outer_rectangle[0]+ outer_rectangle[2])), int(outer_rectangle[1]))

    area_of_interest_corners = (topLeft, topRight, bottomLeft, bottomRight)

    if show_area_of_interest:
        cv2.imshow('Area of interest for hole detection', raw_frame)
        cv2.waitKey(0)

    return raw_frame, area_of_interest_corners


def move_area_of_interest(area_of_interest_corners, x_shift_pix, y_shift_pix):
    shifted_aoi = []
    for point in area_of_interest_corners:
        shifted_point = list(point)
        shifted_point[0] += x_shift_pix
        shifted_point[1] += y_shift_pix
        shifted_aoi.append(tuple(shifted_point))

    return tuple(shifted_aoi)





