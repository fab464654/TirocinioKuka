
# This file contains calibration data for the KukaCam. All the parameters were copied from "KukaCam.yaml" file

import numpy as np

distCoeff = np.zeros((4, 1), np.float64)  # Camera distortion coefficients
# k1<0 to remove barrel distortion
k1, k2, p1, p2 = -0.170962, 0.193219, 0.002225, -0.000648  #copied from KukaCam.yaml
distCoeff[0, 0], distCoeff[1, 0], distCoeff[2, 0], distCoeff[3, 0] = k1, k2, p1, p2

# Camera matrix copied from KukaCam.yaml
cam = np.matrix([[1372.38754,         0., 777.61866],
                 [0., 1373.25601, 575.65429],
                 [0.,         0.,         1.]])
