

def position_controller(considered_center, radius_center_circle, center_threshold, frame_width, frame_height, show_prints):
    camera_center_x = int(frame_width  / 2)
    camera_center_y = int(frame_height / 2)

    camera_endeff_displacement_x = 0  # pixels along x ????
    camera_endeff_displacement_y = 0  # pixels along y ????

    x_error_pix = considered_center[0] - (camera_center_x - camera_endeff_displacement_x)
    y_error_pix = considered_center[1] - (camera_center_y - camera_endeff_displacement_y)

    image_dpi = 60
    inch_to_cm = 2.54

    if abs(x_error_pix) > center_threshold:
        x_ref_cm = x_error_pix / image_dpi * inch_to_cm  #move along x of "x_ref_cm" centimeters

    if abs(y_error_pix) > center_threshold:
        y_ref_cm = y_error_pix / image_dpi * inch_to_cm  #move along y of "y_ref_cm" centimeters

    if show_prints:
        print("[Alignment...] error in pixels (x,y):          (", x_error_pix,y_error_pix, ")")
        print("               error in 'equivalent' cm (x,y): (", x_ref_cm,y_ref_cm, ")")


    return x_ref_cm, y_ref_cm, x_error_pix, y_error_pix

