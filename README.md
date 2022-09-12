# TirocinioKuka
This repository contains the developed code during my 6-CFU traineeship at UniVR. It's part of a project aimed at recognizing holes on a 3D printed gadget and insert screws in them using the Kuka Robot.

_Initial project objectives:_
- Given a plastic 3D printed gadget, localize it, find the threaded holes;
- Once the centers are detected, map 2D pixels into 3D coordinates;
- Plan the Kuka Robot motion accordingly to insert screws into the threaded holes.

In the following, a brief explanation of the (last) developed code:_
1) Read the camera frame, find the area of interest (base with holes)
STEP 2) Detect the holes inside the area of interest
STEP 3) Filter the detected holes:
STEP 4) Make a graph from the pixels' coordinates and find the best path for the robot to follow (optional)
STEP 5) Try to map 2D pixel coordinates into 3D points using camera parameters matrices
STEP 6) TO DO-CHECK if reliable: Control the robot in order to reach the center first hole
        Maybe try to use the above 2D->3D projections, if correct; probably a second alignment process is required afterwards
