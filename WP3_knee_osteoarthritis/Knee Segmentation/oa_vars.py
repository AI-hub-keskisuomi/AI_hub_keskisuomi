import math
import numpy as np
import cv2 as cv
from sklearn.linear_model import LinearRegression

def get_axel(arr: list, axis: int = 0):
    """Helper function for getting one axel from a list of y,x-coordinates

    Args:
        arr (list): List that has coordinates in the format (y,x)
        axis (int, optional): axis 0 returns the x-axis, any other value
         returns the y-axis. Defaults to 0.

    Returns:
        list: List of integers containing the x or y coordinates of the original
        list
    """
    axels = list(zip(*arr))
    return list(axels[0 if axis == 1 else 1])
def adjust_eminentia_angle(tibia_coords, em_ind, lateral = False, interv = 20):
    """Used for testing the amount of pixels to use for calculating the linear 
       models for eminentia-angle calculation relying on the trackbar-
       functionality of opencv.

    Args:
        tibia_coords (list[tuple]): coordinates of the tibial plane in the 
        eminentia region
        em_ind (int): index of the coordinate of the eminentia in tibia_coords
        lateral (bool, optional): Whether the eminentia to be assessed is on the 
        lateral side. Defaults to False.
        interv (int, optional): Initial amount of pixels to consider. 
        Defaults to 20.
    """
    min_x = min(get_axel(tibia_coords))
    min_y = min(get_axel(tibia_coords,1))
    scaled_coords = [(y-min_y,x-min_x) for y,x in tibia_coords]
    #max_x = max(get_axel(scaled_coords))
    max_y = max(get_axel(scaled_coords,1))

    def get_models(coords,ind,interv,lateral):
        pts = coords[max(ind - interv, 0):ind]
        prev_k = LinearRegression().fit(np.array(get_axel(pts)).reshape((-1, 1)), 
                                        np.array(get_axel(pts, axis=1)))
        pts = coords[ind : min(ind +  interv, len(coords))]
        next_k = LinearRegression().fit(np.array(get_axel(pts)).reshape((-1, 1)), 
                                        np.array(get_axel(pts, axis=1)))
        return (prev_k.coef_[0], prev_k.intercept_, 
                next_k.coef_[0], next_k.intercept_)

    def on_trackbar(interv = interv):
        k1,c1,k2,c2 = get_models(scaled_coords,em_ind,interv,lateral) 
        img= np.zeros((max_y + 1, 500, 3))
        for coord in scaled_coords:
            img[coord] = (255,255,255)
        for x in range(0, img.shape[1]):
            y = int(k1*x + c1)
            if 0 <= y < img.shape[0]:
                img[y,x] = [255,0,0]
            y = int(k2*x + c2)
            if 0 <= y < img.shape[0]:
                img[y,x] = [0,255,0]
        cv.imshow("", img)

    on_trackbar()
    # Name of trackbar, window-name, initial value, max value, callback-function
    cv.createTrackbar("pixels to consider", "", interv, 50, on_trackbar)
    cv.waitKey()


def find_eminentia(tibia_coords: list, em_loc: tuple, 
                   tibial_plane_model: LinearRegression,
                   pixel_spacing = [.2, .2], scale=1., 
                   lateral = False, interv = 20) -> dict:
    """Locates the eminentia and calculates its height and angle

    Args:
        tibia_coords (list): coordinates of the tibial plane in the image as a
        list of (y,x) coordinates
        em_loc (tuple): The pair of  (y,x) coordinates between which the 
        eminentia is expected to reside
        tibial_plane_model (LinearRegression): The linear regression model 
        modeling the tibial plane outside of the eminentia region
        pixel_spacing (list, optional): Pixelspacing-variable of the 
        original DICOM-image. Defaults to [.2, .2].
        scale (float, optional): The possible downscaling-factor of the original 
        image. Defaults to 1. i.e. no downscaling.
        lateral (bool, optional): Whether you are trying to locate the lateral 
        eminentia. Defaults to False.
        interv (int, optional): Amount of pixels to consider when calculating the 
        linear models for the calculation of the angle of the eminentia. 
        Defaults to 20.

    Returns:
        dict: The location of the eminentia along with the calculated variables
    """
    variables={}
    em_a, em_b = em_loc
    coords = sorted([coord for coord in tibia_coords 
                     if coord[1] > em_a + 6 and coord[1] < em_b - 6], 
                     key=lambda x: (1 if lateral else -1) * x[1])
    
    ind = 5
    ref = np.mean(get_axel(coords, axis = 1))
    for i in range(5, len(coords) - 5):
        if coords[i + 5][0] - coords[i - 5][0] >= 0 and coords[i][0] < ref:
            ind = i
            break
    # adjust_eminentia_angle(coords,ind,lateral,interv)
    if ind < len(coords) - 5:
        variables['height'] = (tibial_plane_model.predict([[coords[ind][1]]])[0]
                               - coords[ind][0]) * pixel_spacing[1] / scale
        pts = coords[max(ind - interv, 0):ind]
        prev_k = LinearRegression().fit(
            np.array(get_axel(pts)).reshape((-1, 1)), 
            np.array(get_axel(pts, axis=1))
            ).coef_[0] * (1 if lateral else -1)
        pts = coords[ind : min(ind + interv, len(coords))]
        next_k = LinearRegression().fit(
            np.array(get_axel(pts)).reshape((-1, 1)), 
            np.array(get_axel(pts, axis=1))
            ).coef_[0] * (1 if lateral else -1)
        if prev_k * next_k == -1:
            ang = 90
        else:
            ang = math.atan((prev_k - next_k) 
                            / (1 + prev_k * next_k)) * 180 / math.pi
        if ang < 0:
            ang = 180 + ang

        variables['angle'] = ang
        variables['x'] = coords[ind][1]
    else:
        variables['height'] = ""
        variables['angle'] = ""
        variables['x'] = ""
    return variables

def calculate_vars(tibia_coords: list, femur_coords: list, pixel_spacing: list,
                   ignore_edge = 10) -> dict:
    """Calculate different variables linked with knee osteoarthritis from the 
       found coordinates of the tibial and femoral planes

    Args:
        tibia_coords (list): List of (y,x) coordinates containing the 
        location of the tibia
        femur_coords (list): List of (y,x) coordinates containing the 
        locatioon of the femur
        pixel_spacing (list): PixelSpacing-feature of the original 
        DICOM-image. Used for calculating joint space width as mm.
        ignore_edge (int, optional): How many pixels to ignore at 
        the edges of the bones as they can be a bit noisy. 
        Defaults to 10.

    Returns:
        dict: Dict containing all of the calculated variables
    """
    tibia_coords = sorted(tibia_coords, key = lambda x: x[1])
    femur_coords = sorted(femur_coords, key = lambda x: x[1])
    tibia_coords = tibia_coords[ignore_edge : -ignore_edge]
    femur_coords = femur_coords[ignore_edge : -ignore_edge]

    full_tibia_model = LinearRegression().fit(
        np.array(get_axel(tibia_coords)).reshape((-1, 1)),
        np.array(get_axel(tibia_coords, axis=1))
    )

    a = min(get_axel(tibia_coords))
    b = max(get_axel(tibia_coords))
    for coord in tibia_coords:
        if (coord[0] < full_tibia_model.predict([[coord[1]]])[0] and
            coord[1] > a + 40):
            em_a = (coord[1])
            break
    for coord in tibia_coords[::-1]:
        if (coord[0] < full_tibia_model.predict([[coord[1]]])[0] and 
            coord[1] < b - 40):
            em_b = (coord[1])
            break
    mjsw = []
    ljsw = []
    for y, x in tibia_coords:
        if em_a <= x <= em_b:
            continue
        # very inefficient way to do this, as there is ever only one of these, 
        # couldn't be arsed to do it properly though as the numbers are small enough
        femur_y = [fy for fy, fx in femur_coords if fx == x][0]
        if x < em_a:
            ljsw.append((y - femur_y) * pixel_spacing[1])
        else:
            mjsw.append((y - femur_y) * pixel_spacing[1])
    reg_coords = [(y, x) for y, x in tibia_coords if x <= em_a or x >= em_b]
    xs = np.array(get_axel(reg_coords)).reshape((-1, 1))
    tibial_plane = LinearRegression().fit(
        xs, np.array(get_axel(reg_coords, axis = 1))
    )
    interv = int(2 / pixel_spacing[1])
    mem=find_eminentia(tibia_coords, em_loc = (em_a, em_b), tibial_plane_model=tibial_plane,
                       pixel_spacing = pixel_spacing, interv = interv)
    lem=find_eminentia(tibia_coords, em_loc = (em_a, em_b), tibial_plane_model=tibial_plane,
                       pixel_spacing = pixel_spacing, lateral=True, interv = interv)
    return {
        "l_em_height": lem["height"],
        "m_em_height": mem["height"],
        "l_em_angle": lem["angle"],
        "m_em_angle": mem["angle"],
        "l_em_x": lem["x"],
        "m_em_x": mem["x"],
        "l_min_jsw": np.min(ljsw),
        "m_min_jsw": np.min(mjsw),
        "l_avg_jsw": np.mean(ljsw),
        "m_avg_jsw": np.mean(mjsw),
        "tibia_model": [tibial_plane.coef_[0], tibial_plane.intercept_]
    }