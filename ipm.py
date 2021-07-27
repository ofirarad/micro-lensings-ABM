import numpy as np
import time
import aux_functions as af


def crit_curves(centers, A0, dhA, dvA, h_size, v_size):
    # returning a list of slopes and intercept terms for the critical curves linear-approximation,
    # for given cells of size (h_size X v_size). All measurements are in arc-sec
    slopes = -dhA / dvA
    h0 = centers[:, 0].reshape(slopes.shape)
    v0 = centers[:, 1].reshape(slopes.shape)
    intercepts = -h0 * slopes + v0 - (A0 / dvA)
    pplus = (v0 + v_size / 2 - intercepts) / slopes
    pminus = (v0 - v_size / 2 - intercepts) / slopes
    h1 = (slopes > 0) * np.maximum(pminus, h0 - h_size / 2) + (slopes < 0) * np.maximum(pplus, h0 - h_size / 2)
    h2 = (slopes > 0) * np.minimum(pplus, h0 + h_size / 2) + (slopes < 0) * np.minimum(pminus, h0 + h_size / 2)
    v1 = slopes * h1 + intercepts
    v2 = slopes * h2 + intercepts
    return np.hstack((h1, v1, h2, v2))


def ipm_method(state):
    """
    The main routine for running the gravitational lensing simulation using the Inverse Polygon Method (Mediavilla 2006)
    :param state: a dictionary containing all variables of the lens and user definitions
    :return: beta_grid_h, beta_grid_v, mu_grid, nlin_tiles
    """
    # First, importing all variables from the dictionary 'state'
    theta_ein2cm = state['theta_ein2cm']
    beta_boundary = state['beta_boundary']
    beta_res = state['beta_res']
    epsilon = state['epsilon']
    mu_h = state['mu_h']
    mu_v = state['mu_v']
    num_of_tiles = state['num_of_tiles']
    m = state['m']
    zeta = state['zeta']
    max_memory = state['max_memory']
    low_log_mass = state['low_log_mass']
    Mtypical = state['Mtypical']
    IPM_l_parameter = state['IPM_l_parameter']

    pixel2cm = theta_ein2cm * beta_boundary / beta_res  # size of 1 pixel in cm in the source plane
    print('The physical size of 1 pixel is ' + str(beta_boundary / beta_res) + ' Einstein radii\nor ' + str(
        np.format_float_scientific(pixel2cm, 2)) + ' cm in the source plane\n')

    theta_boundaries = [epsilon * mu_h * beta_boundary / 2,
                        epsilon * mu_v * beta_boundary / 2]
    if num_of_tiles == 0:
        # choose the tiles size ~ the scale of lowest theta_ein in the mass distribution
        num_of_tiles_1 = int(
            ((epsilon * mu_v * beta_boundary) / (np.sqrt(10 ** low_log_mass / Mtypical) / IPM_l_parameter)) ** 2)
        # choose the tiles size ~ the scale of source plane pixels
        num_of_tiles_2 = int(beta_res ** 2 * epsilon ** 2 * mu_h * mu_v)
        # choose the tiles size ~ the minimum of the two above options
        num_of_tiles = min(num_of_tiles_1, num_of_tiles_2)
        state['num_of_tiles'] = num_of_tiles
        tile_size = (epsilon * mu_v * beta_boundary) / np.sqrt(num_of_tiles)
    else:
        tile_size = (epsilon * mu_v * beta_boundary) / np.sqrt(num_of_tiles)
    print('Using ' + str(int(np.sqrt(num_of_tiles))) + ' tiles per side, of typical length ' + str(
        tile_size) + ' Einstein radii')
    print('.........Tessellating image plane')
    start_time = time.time()
    theta_vertices, theta = af.tessellate_area(theta_boundaries, np.array([0, 0]), num_of_tiles)
    tiles = retrieve_polygons(theta_vertices)
    print('Finished Tessellating  image plane in ' + str(time.time() - start_time) + 's')

    print('.........Transforming tiles vertices')
    l_tmp = int(max_memory / m.shape[0] * 10 ** 9 / 8)  # the maximum number of images to vector-compute
    s = max(int(theta_vertices.shape[0] / l_tmp), 1)  # the number of sub arrays to vector-compute
    print('Max memory for array: ' + str(l_tmp * m.shape[0] * 8 / 10 ** 9) + 'GB')
    start_time = time.time()
    beta = np.zeros(theta_vertices.shape)

    for i in range(s):
        # Defining images locations theta
        beta[i * l_tmp:(i + 1) * l_tmp] = af.img2src(theta_vertices[i * l_tmp:(i + 1) * l_tmp], m, zeta, state)
        temp_t = time.time() - start_time
        if i % (max(1, int(s / 10))) == 0:
            print('Finished ' + str(round((i + 1) * 100 / s, 2)) + '% in ' + str(round(temp_t)) +
                  's; ~' + str(round(temp_t * (s / (i + 1) - 1))) + 's remaining')
    if s * l_tmp < theta_vertices.shape[0]:  # if some values are left
        beta[s * l_tmp:] = af.img2src(theta_vertices[s * l_tmp:], m, zeta, state)
        print('Totally finished  in ' + str(time.time() - start_time) + 's')
    else:
        print('Totally finished  in ' + str(time.time() - start_time) + 's')

    print('.........Transforming tiles centers to the source plane')
    l_tmp = int(max_memory / m.shape[0] * 10 ** 9 / 8)  # the maximum number of images to vector-compute
    s = max(int(theta.shape[0] / l_tmp), 1)  # the number of sub arrays to vector-compute
    print('Max memory for array: ' + str(l_tmp * m.shape[0] * 8 / 10 ** 9) + 'GB')
    start_time = time.time()
    dA = np.zeros((theta.shape[0], 3))
    for i in range(s):
        _, dA[i * l_tmp:(i + 1) * l_tmp] = af.img2src(theta[i * l_tmp:(i + 1) * l_tmp], m, zeta, state, mu_flag=True)
        temp_t = time.time() - start_time
        if i % (max(1, int(s / 10))) == 0:
            print('Finished ' + str(round((i + 1) * 100 / s, 2)) + '% in ' + str(round(temp_t)) +
                  's; ~' + str(round(temp_t * (s / (i + 1) - 1))) + 's remaining')
    if s * l_tmp < theta.shape[0]:  # if some values are left
        _, dA[s * l_tmp:] = af.img2src(theta[s * l_tmp:], m, zeta, state, mu_flag=True)
        print('Totally finished  in ' + str(time.time() - start_time) + 's')
    else:
        print('Totally finished  in ' + str(time.time() - start_time) + 's')

    print('.........Retrieving tiles in the source plane')
    source_tiles = retrieve_polygons(beta)

    print('.........Binning tiles in the source plane pixels-grid')
    start_time = time.time()
    beta_grid_h, beta_grid_v, mu_grid, nlin_tiles = polygons_binning(source_tiles, tiles, dA, state)
    print('Finished binning in ' + str(time.time() - start_time) + 's')
    return beta_grid_h, beta_grid_v, mu_grid, nlin_tiles, beta, theta


def poly_pixel_overlap(polygon, h_min, v_min, d_h, d_v, no_pixel=False):
    """
    v_min and h_min are the bottom and left boundaries of the pixel.
    d_h and d_v are the horizontal and vertical size of 1 pixel
    polygon is a numpy array of size (q,2) where each row represents the horizontal and vertical coordinates of a vertex

    :param polygon:
    :param h_min:
    :param v_min:
    :param d_h:
    :param d_v:
    :param no_pixel:
    :return:
    """

    lines = tetra_sides(polygon)  # gives a list of numpy arrays containing the slope,intercept,initial and final
    # - horizontal points of each line of a convex tetragon
    if no_pixel:
        h_min = np.min(polygon[:, 0])
        v_min = np.min(polygon[:, 1])
        h_max = np.max(polygon[:, 0])
        v_max = np.max(polygon[:, 1])
        d_h = h_max - h_min
        d_v = v_max - v_min
    else:
        v_max = v_min + d_v
        h_max = h_min + d_h
    s_tot = 0

    if h_max < np.min(polygon[:, 0]) or h_min > np.max(polygon[:, 0]) \
            or v_max < np.min(polygon[:, 1]) or v_min > np.max(polygon[:, 1]):
        # If pixel if out of maximal perimeter
        return 0

    for l in lines:
        m = l[0]
        n = l[1]
        x1 = l[2]
        x2 = l[3]
        if x1 < x2:
            orientation = 1
        else:
            orientation = -1
            x1, x2 = x2, x1  # x1 should always be smaller than x2 for this algorithm
        if x1 == x2:
            s = 0  # if it is a vertical line
        elif x2 < h_min or x1 > h_max:
            s = 0
        elif m > 0:
            p_plus = (v_max - n) / m  # the intersection of the line with the pixel upper boundary v_max
            p_minus = (v_min - n) / m  # the intersection of the line with the pixel lower boundary v_min
            if p_plus < h_min:
                s = d_v * (min(x2, h_max) - max(x1, h_min))
            elif p_minus > h_max:
                s = 0
            else:
                low_int = max(x1, p_minus, h_min)  # the lower and upper boundaries of the integral
                up_int = min(x2, p_plus, h_max)
                s = m / 2 * (up_int ** 2 - low_int ** 2) + (n - v_min) * (up_int - low_int) + d_v * (
                        min(x2, h_max) - up_int)
        elif m < 0:
            p_plus = (v_max - n) / m  # the intersection of the line with the pixel upper boundary h_max
            p_minus = (v_min - n) / m  # the intersection of the line with the pixel lower boundary h_min
            if p_plus > h_max:
                s = d_v * (min(x2, h_max) - max(h_min, x1))
            elif p_minus < h_min:
                s = 0
            else:
                low_int = max(x1, p_plus, h_min)  # the lower and upper boundaries of the integral
                up_int = min(x2, p_minus, h_max)
                s = m / 2 * (up_int ** 2 - low_int ** 2) + (n - v_min) * (up_int - low_int) + d_v * (
                        low_int - max(x1, h_min))
        elif m == 0:
            if n > v_max:
                s = d_v * (min(x2, h_max) - max(h_min, x1))
            elif n < v_min:
                s = 0
            else:
                s = (n - v_min) * (min(x2, h_max) - max(h_min, x1))
        s_tot += orientation * s
    return abs(s_tot)


def polygons_binning(source_polygons, image_polygons, dA, state):
    """

    :param source_polygons:
    :param image_polygons:
    :param dA:
    :param state: a dictionary containing all variables of the lens and user definitions
    :return:
    """
    m = state['m']
    zeta = state['zeta']
    IRS_nlin_tiles = state['IRS_nlin_tiles']
    boundary = state['beta_boundary']
    res = state['beta_res']
    # The boundaries for the source-plane FOV
    max_h = boundary / 2
    max_v = boundary / 2
    min_h = -boundary / 2
    min_v = -boundary / 2
    grid_vec_h, h_step = np.linspace(min_h, max_h, res, retstep=True)
    grid_vec_v, v_step = np.linspace(min_v, max_v, res, retstep=True)
    beta_grid_h, beta_grid_v = np.meshgrid(grid_vec_h, grid_vec_v)
    mu_tot = np.zeros(beta_grid_h.shape)

    non_lin_polygons = []
    # the angular area of a single polygons in the image plane, assuming tiles of same size
    s_img = poly_pixel_overlap(image_polygons[0], 0, 0, 0, 0, no_pixel=True)
    tile_size = np.sqrt(s_img)  # assuming a rectangular tile
    start_time = time.time()
    # The non-linear tiles criterion, from Mediavilla (2006)
    non_crit_cells = np.squeeze((np.abs(dA[:, 1]) + np.abs(dA[:, 2])) * tile_size / 100 < np.abs(dA[:, 0]))
    start_time = time.time()
    for i, polygon in enumerate(source_polygons):
        if non_crit_cells[i]:
            s_src = poly_pixel_overlap(polygon, 0, 0, 0, 0,
                                       no_pixel=True)  # the angular area of the polygon in the source plane
            avg_mu = s_img / s_src  # the average polygon magnification
            p_min_h = np.min(polygon[:, 0])  # the most left vertical line
            p_min_v = np.min(polygon[:, 1])  # the lowest horizontal line
            p_max_h = np.max(polygon[:, 0])  # the most right vertical line
            p_max_v = np.max(polygon[:, 1])  # the highest horizontal line
            for j, pixel_h in enumerate(grid_vec_h):
                # pixel_h is the angular horizontal value of the j-th column
                if p_min_h - 1 * h_step <= pixel_h <= p_max_h + 0 * h_step:  # to only calculate the overlap with relevant pixels
                    for k, pixel_v in enumerate(grid_vec_v):
                        # pixel_v is the angular horizontal value of the k-th row
                        if p_min_v - 1 * v_step <= pixel_v <= p_max_v + 0 * v_step:  # to only calculate the overlap with relevant pixels
                            # the area of the polygon-pixel overlap in the source plane
                            # normalized by the average cell magnification; Eq.(21) Mediavilla (2011)
                            mu_tot[j, k] += avg_mu * poly_pixel_overlap(polygon, pixel_h, pixel_v, h_step,
                                                                        v_step)

        else:
            if IRS_nlin_tiles > 1:
                # Treat non-linear cells using IRS method
                image_polygon = image_polygons[i]
                tmp_boundary = [np.max(image_polygon[:, 0]) - np.min(image_polygon[:, 0]),
                                np.max(image_polygon[:, 1]) - np.min(image_polygon[:, 1])]
                tmp_offset = [(np.max(image_polygon[:, 0]) + np.min(image_polygon[:, 0])) / 2,
                              (np.max(image_polygon[:, 1]) + np.min(image_polygon[:, 1])) / 2]
                tmp_theta, _ = af.tessellate_area(tmp_boundary, tmp_offset,
                                                      int((np.sqrt(IRS_nlin_tiles) - 1) ** 2))
                # Transforming the non-linear rays to the source plane
                sub_tiles_beta = af.img2src(tmp_theta, m, zeta, state)
                for k, tmp_beta in enumerate(sub_tiles_beta):
                    idx_h = int((tmp_beta[0] - grid_vec_h[0]) // h_step)  # the horizontal index of the relevant ray
                    idx_v = int((tmp_beta[1] - grid_vec_v[0]) // v_step)  # the horizontal index of the relevant ray
                    if idx_h >= 0 and idx_v >= 0 and idx_h < res and idx_v < res:
                        mu_tot[idx_h, idx_v] += s_img / (IRS_nlin_tiles)  # simple IRS method
                        # normalized by the average cell magnification; Eq.(21) Medevilla (2011)
            else:
                non_lin_polygons.append(i)
        temp_t = time.time() - start_time
        if i % (max(1, int(len(source_polygons) / 100))) == 0:
            print('Finished ' + str(round((i + 1) * 100 / len(source_polygons), 4)) + '% in ' + str(round(temp_t)) +
                  's; ~' + str(round(temp_t * (len(source_polygons) / (i + 1) - 1))) + 's remaining')
    return beta_grid_h, beta_grid_v, mu_tot.T / (h_step * v_step), non_lin_polygons


def retrieve_polygons(theta_vert):
    """
    returns a list of polygons coordinates, ordered from bottom left to top right, horizontal then vertical
    :param theta_vert: the complete list of vertices coordinates ordered from bottom left to top right, horizontal then vertical
    :return: polygons: the list of numpy arrays of coordinates of each polygon's vertices v1->v2->v3->v4 (counter clockwise)
    """
    tiles_per_dim = int(np.sqrt(theta_vert.shape[0]) - 1)
    polygons = []
    for i in range(tiles_per_dim):
        for j in range(tiles_per_dim):
            v1 = i * (tiles_per_dim + 1) + (j)
            v2 = i * (tiles_per_dim + 1) + (j + 1)
            v3 = (i + 1) * (tiles_per_dim + 1) + (j + 1)
            v4 = (i + 1) * (tiles_per_dim + 1) + (j)
            pol_vertices = theta_vert[[v1, v2, v3, v4], :]
            polygons.append(pol_vertices)
    return polygons


def tetra_sides(polygon):
    """

    :param polygon:
    :return:
    """
    l = []
    num_ver = polygon.shape[0]
    for i in range(num_ver):
        l.append(vertices2line(polygon[i], polygon[(i + 1) % num_ver]))
    return l


def vertices2line(ver1, ver2):
    """

    :param ver1:
    :param ver2:
    :return:
    """
    eps = 10 ** (-16)
    x1 = ver1[0]
    x2 = ver2[0]
    y1 = ver1[1]
    y2 = ver2[1]
    if y1 == y2:
        m = 0
        n = y1
    elif x1 == x2:
        m = 1 / eps
        n = 0
    else:
        m = (y2 - y1) / (x2 - x1)
        n = (y1 * x2 - y2 * x1) / (x2 - x1)
    return np.array([m, n, x1, x2])



