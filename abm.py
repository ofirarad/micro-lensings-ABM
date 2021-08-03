import numpy as np
import time
from scipy.signal import find_peaks
import aux_functions as af


def abm_light_curve(path_vec, theta_lim, beta_lim, n_pixels, eta, num_iter, delta_beta, state):
    """
    Performs the calculation of the light curve based on the given path path_vec.
    :param path_vec: an (n_steps,2) array containing n_steps sets of coordinates for the path
    :param theta_lim: Initial image plane half-boundaries for the AOP algorithm;in Einstein radii;[horizontal, vertical]
    :param beta_lim: Initial source plane half-boundaries for the AOP algorithm;in Einstein radii; scalar
    :param n_pixels: The image plane division factor of the AOP algorithm
    :param eta: The source plane division factor of the AOP algorithm
    :param num_iter: The number of divisions of the AOP algorithm
    :param delta_beta: The final source plane half-boundary for the AOP algorithm; scalar - same boundary for both axis
    :param state: A dictionary containing essential parameters of the script
    :return: A numpy array of magnification values (true count) of each step in the given path
    """
    # unique image plane area of each ray
    s_img = 4 * theta_lim[0] * theta_lim[1] / n_pixels ** (2 * np.floor(num_iter)+2)
    kernel_flag = state['kernel_flag']  # whether or nor to apply Gaussian kernel
    if kernel_flag:
        kernel = gaussian_kernel(3, 9)  # 9X9 matrix, corresponding to a range of +-3*sigma of a Gaussian kernel
    light_curve = []
    for i, path_point in enumerate(path_vec):
        start_time = time.time()
        theta_rays, beta_rays = abm_single_routine(theta_lim, [0, 0], beta_lim, path_point, n_pixels, eta, 0, num_iter,
                                                   state)
        if beta_rays.shape[0] > 1:  # if there are rays in this source plane region
            if kernel_flag:  # to apply Gaussian profile
                #  bin the source plane area of +- delta_beta, with 9^2 bins
                _, _, bins_mag = af.mag_binning(beta_rays, s_img, 2 * delta_beta, 9, path_point)
                #  weigh bins using Gaussian profile of +- 3 half widths
                light_curve.append(np.sum(kernel * bins_mag))
            else:  # not to apply Gaussian profile
                light_curve.append(beta_rays.shape[0] * s_img / (np. pi * delta_beta ** 2))
        else:  # if no rays in this source plane region
            light_curve.append(0)
        print('Finished ' + str(100 * (i + 1) / path_vec.shape[0]) + '% in ' + str(time.time() - start_time) + 's')
    return np.array(light_curve)


def abm_method(state):
    """
    The main routine for running the gravitational lensing simulation using the Adaptive Boundary Method
    :param state: a dictionary containing all variables of the lens and user definitions
    :return: beta_grid_h, beta_grid_v, mu_grid, nlin_tiles
    """
    # First, importing all variables from the dictionary 'state'
    theta_ein2cm = state['theta_ein2cm']
    beta_boundary = state['beta_boundary']
    epsilon = state['epsilon']
    mu_h = state['mu_h']
    mu_v = state['mu_v']
    source_size = state['source_size']
    n_pixels = state['n_pixels']
    n_steps = state['n_steps']
    vt_ein = state['vt_ein']

    # the image plane boundaries corresponding to the source plane FOV
    theta_boundaries = [epsilon * mu_h * beta_boundary / 2, epsilon * mu_v * beta_boundary / 2]
    beta_boundary_0 = state['beta_0'] * beta_boundary  # initial source plane radial-boundary for the AOP algorithm
    state['plot_flag'] = False
    delta_beta = source_size / theta_ein2cm  # final source plane half boundary in Einstein radii
    # eta: the factor by which the initial beta boundary around a source point is divided in each iteration
    eta = n_pixels * state['eta_ratio']
    state['eta'] = eta
    # The number of iterations, note it is a real number, and not an integer
    num_iter = 1 + np.log(beta_boundary_0 / delta_beta) / np.log(eta)
    state['num_iter'] = num_iter
    print('Image plane division factor is ' + str(n_pixels))
    print('Source plane boundary division factor is ' + str(eta))
    print('The target source plane resolution is ' + str(source_size / theta_ein2cm) + ' Einstein radii')
    print('Number of iterations is '+str(num_iter))

    print('Creating shear-parallel light curve')
    start_time = time.time()
    path_vec = np.linspace(-beta_boundary / 2, beta_boundary / 2, num=n_steps, endpoint=True)
    path_vec = np.vstack((path_vec, np.zeros((1,path_vec.shape[0])))).T
    # calculating the light curve based on the path vector
    light_curve = abm_light_curve(path_vec, theta_boundaries, beta_boundary_0, n_pixels, eta, num_iter, delta_beta,
                                  state)
    print('Totally finished in ' + str(time.time() - start_time))
    print('Refining parallel light curve path')
    start_time = time.time()
    time_steps, light_curve = refine_light_curve(path_vec, light_curve, vt_ein, theta_boundaries, beta_boundary_0,
                                                 n_pixels, eta, num_iter, delta_beta, state)
    print('Totally finished in ' + str(time.time() - start_time))


    print('Creating shear-perpendicular light curve')
    start_time = time.time()
    path_vec = np.linspace(-beta_boundary / 2, beta_boundary / 2, num=n_steps, endpoint=True)
    path_vec = np.vstack((np.zeros((1, path_vec.shape[0])), path_vec)).T
    # calculating the light curve based on the path vector
    light_curve_perp = abm_light_curve(path_vec, theta_boundaries, beta_boundary_0, n_pixels, eta, num_iter, delta_beta,
                                       state)
    print('Totally finished in ' + str(time.time() - start_time))
    print('Refining perpendicular light curve path')
    start_time = time.time()
    time_steps_perp, light_curve_perp = refine_light_curve(path_vec, light_curve_perp, vt_ein, theta_boundaries,
                                                           beta_boundary_0, n_pixels, eta, num_iter, delta_beta, state)
    print('Totally finished in ' + str(time.time() - start_time))
    return time_steps, light_curve, time_steps_perp, light_curve_perp


def abm_single_routine(theta_boundaries, theta_offset, beta_boundaries, beta_offset, n_pixels, eta, itr_n, max_itr,
                       state):
    """
    This is a recursive method, to start the method from initial conditions make sure itr_n=0.
    This is the ABM routine for a single point in the source plane. For each iteration, of the max_itr, an image plane
    area is divided to n_pixels^2, while the source plane area is reduced by a factor eta^2. Only those image plane
    pixels that map within the source plane boundary continue the recursive algorithm. If this is the last iteration,
    the source plane division factor is reduced to fit the final desired source plane resolution.
    :param theta_boundaries: The image plane half boundaries ([horizontal,vertical] - rectangular area)
    :param theta_offset: The center coordinates of the image plane area ([horizontal,vertical])
    :param beta_boundaries: The radius of the source plane area (circular region)
    :param beta_offset: The center coordinates of the source plane area ([horizontal,vertical])
    :param n_pixels: The number of pixels to divide the image plane area in each axis
    :param eta: The source plane scale division factor
    :param itr_n: The number of current iteration.
    :param max_itr: The maximum number of iterations, a real number, determined by the final source plane resolution
    :param state: A dictionary containing essential parameters of the script
    :return:
    """
    n1 = int(np.floor(max_itr))  # the integer part of the number of iterations
    n2 = max_itr - n1  # the remainder of the real number of iterations
    rel_pixels, rel_pixels_beta = split_and_identify(theta_boundaries, theta_offset, beta_boundaries, beta_offset,
                                                     n_pixels, state)
    new_theta_boundaries = [theta_boundaries[0] / n_pixels, theta_boundaries[1] / n_pixels]
    total_rel_pixel = []
    total_rel_pixel_beta = []
    if itr_n < n1 - 1:  # if didn't reach the final division
        new_beta_boundaries = beta_boundaries / eta  # decrease the source plane boundary by a factor eta
        for rel_pixel in rel_pixels:
            temp_pixels, temp_pixels_beta = abm_single_routine(new_theta_boundaries, rel_pixel, new_beta_boundaries,
                                                               beta_offset, n_pixels, eta, itr_n + 1, max_itr, state)
            if len(temp_pixels_beta) > 0:
                total_rel_pixel.append(temp_pixels)
                total_rel_pixel_beta.append(temp_pixels_beta)
    elif itr_n == n1 - 1:  # if reached the final division (where eta is effectively smaller, because of a power of n2)
        new_beta_boundaries = beta_boundaries / eta ** n2  # decrease the source plane boundary by a factor eta^n2
        for rel_pixel in rel_pixels:
            temp_pixels, temp_pixels_beta = abm_single_routine(new_theta_boundaries, rel_pixel, new_beta_boundaries,
                                                               beta_offset, n_pixels, eta, itr_n + 1, max_itr, state)
            if len(temp_pixels_beta) > 0:
                total_rel_pixel.append(temp_pixels)
                total_rel_pixel_beta.append(temp_pixels_beta)
    elif len(rel_pixels) > 0:  # finished dividing source plane, reached desired boundary. Collect relevant pixels
        total_rel_pixel.append(rel_pixels)
        total_rel_pixel_beta.append(rel_pixels_beta)
    return np.array([item for sublist in total_rel_pixel for item in sublist]), \
           np.array([item for sublist in total_rel_pixel_beta for item in sublist])


def gaussian_kernel(num_sigma, num_pixels):
    """
    Returns an array representing a kernel of Gaussian distribution, in the range +-num_sigma * sigma
    :param num_sigma: the number of half_widths in each direction; half-boundary of the kernel
    :param num_pixels: number of pixels for each axis
    :return: a (num_pixels,num_pixels) array of a Gaussian kernel in the region +- num_sigma standard deviations
    """
    temp_h, temp_v = np.meshgrid(np.linspace(-num_sigma, num_sigma, num_pixels),
                                 np.linspace(-num_sigma, num_sigma, num_pixels))
    kernel = np.exp(-(temp_h ** 2 + temp_v ** 2) / 2)  # a Gaussian kernel for a finite point source
    return kernel / np.sum(kernel)


def path2displacement(path_vec):
    """
    Transforms a 2D path into a 1D displacement path (relative to the initial position)
    :param path_vec: an (n>1,2) array containing coordinates of n steps
    :return: an (n) array containing displacement values from first cell of input array
    """
    [h0, v0] = path_vec[0]
    h_delta = path_vec[:,0] - h0
    v_delta = path_vec[:,1] - v0
    return np.sqrt(h_delta ** 2 + v_delta ** 2)


def refine_light_curve(path_vec, light_curve, vt_ein, theta_lim, beta_lim, n_pixels, eta, num_iter,
                       delta_beta, state):
    """
    This function analyzes the light curve path (full coordinates) and magnitude, and return a refined path with more
    sampling points around prominent peaks in the light curve. The function returns light curve vector, as well as the
    complimentary time steps vector, in units of years, containing the input light curve as well as the newly sampled
    points.
    :param path_vec: numpy array containing the coordinates of each step in the input light curve (n_steps,2)
    :param light_curve: numpy array containing the magnification value of each step in the light curve
    :param vt_ein: The transverse velocity in units of Einstein radii per second
    :param theta_lim: Initial image plane half-boundaries for the AOP algorithm;in Einstein radii;[horizontal, vertical]
    :param beta_lim: Initial source plane half-boundaries for the AOP algorithm;in Einstein radii; scalar
    :param n_pixels: The image plane division factor of the AOP algorithm
    :param eta: The source plane division factor of the AOP algorithm
    :param num_iter: The number of divisions of the AOP algorithm
    :param delta_beta: The final source plane half-boundary for the AOP algorithm; scalar - same boundary for both axis
    :param state: A dictionary containing essential parameters of the script
    :return: time steps vector, and light curve vector
    """
    refined_path_vec = refine_time_steps(path_vec,light_curve)
    refined_light_curve = abm_light_curve(refined_path_vec, theta_lim, beta_lim, n_pixels, eta, num_iter,
                                          delta_beta, state)
    # Order both path_vec and refined_path_vec based on displacement from initial point of path_vec
    # time steps in units of years
    time_steps = path2displacement(path_vec) * (1 / vt_ein / 31557600)  # displacement of path_vec
    refined_path_vec = np.insert(refined_path_vec,0,path_vec[0],axis=0)  # adding initial point to refined path_vec
    time_steps_refined = path2displacement(refined_path_vec) * (1 / vt_ein / 31557600)  # displacement refined_path_vec
    time_steps_refined = time_steps_refined[1:]  # omit first element
    time_steps = np.append(time_steps, time_steps_refined)
    light_curve_complete = np.append(light_curve, refined_light_curve)
    light_curve_complete = np.array(light_curve_complete)[np.argsort(time_steps)]
    time_steps = np.sort(time_steps)
    return time_steps, light_curve_complete


def refine_time_steps(path_steps, light_curve_in):
    """
    Returns a numpy array of refined path steps based on peaks in original light curve.
    Uses a peak finding technique "find_peaks" from scipy.signal; Divide the additional time steps proportional to the
    inverse of the width of each detected peak.
    :param path_steps: numpy array of size (n,2) of the n path steps
    :param light_curve_in: numpy array of magnification count per time step
    :return: numpy array of refined path steps, size (n,2)
    """
    light_curve = light_curve_in
    light_curve = np.log10(np.abs(light_curve) + 1)  # the 1 term is to prevent log(0) errors
    n_steps = path_steps.shape[0]
    # find all prominent peaks
    peaks, properties = find_peaks(light_curve, prominence=0.1, width=0)
    # we spread the same number of steps as the original light curve, only between the detected peaks.
    n_peaks = len(peaks) # the number of peaks
    print('Found '+str(n_peaks)+' peaks')
    print(peaks)
    print(properties["widths"])
    print(properties["prominences"])
    tot_width = sum(1 / np.array(properties["widths"]))
    # a list of weights of each peak based on their widths, normalized to the number of steps of the refined path
    weights = [int(1 / tmp_width / tot_width * n_steps) for tmp_width in properties["widths"]]
    # left limit of each peak, in same coordinates as input steps
    left_lims = path_steps[[int(tmp_lim) for tmp_lim in peaks - 2 * (properties["widths"] / 2)], :]
    # right limit of each peak, in same coordinates as input steps
    right_lims = path_steps[[int(tmp_lim) for tmp_lim in peaks + 2 * (properties["widths"] / 2)], :]
    # creating the refined path steps
    new_path_steps_h = []
    new_path_steps_v = []
    for i, peak in enumerate(peaks):
        tmp_steps_h =np.linspace(left_lims[i,0], right_lims[i,0],num=weights[i],endpoint=True)
        tmp_steps_v =np.linspace(left_lims[i,1], right_lims[i,1],num=weights[i],endpoint=True)
        new_path_steps_h = np.append(new_path_steps_h, tmp_steps_h)
        new_path_steps_v = np.append(new_path_steps_v, tmp_steps_v)
    return np.vstack((new_path_steps_h,new_path_steps_v)).T  # to yield an (n,2) array


def split_and_identify(theta_boundaries, theta_offset, beta_lim, beta_offset, n_pixels, state):
    """
    splitting the image plane region given by +-beta_boundary from the center point theta_offset. Then map these rays to
    the source plane and then return those rays that map to and area +-beta_boundary from the center point beta_offset.
    :param theta_boundaries: The image plane half boundaries ([horizontal,vertical] - rectangular area)
    :param theta_offset: The center coordinates of the image plane area ([horizontal,vertical])
    :param beta_lim: The radius of the source plane area (circular region)
    :param beta_offset: The center coordinates of the source plane area ([horizontal,vertical])
    :param n_pixels: The number of pixels to divide the image plane area in each axis
    :param state: A dictionary containing essential parameters of the script
    :return: python list of image plane pixels and python list of mapped source plane pixels, both lists only with
    pixels that map into the given source plane boundary.
    """
    m = state['m']  # The masses of the lens
    zeta = state['zeta']  # The coordinates of each mass in the lens
    # First split the image plane region to n_pixels X n_pixels pixels
    theta_pixels = af.tessellate_area(theta_boundaries, theta_offset, n_pixels ** 2, ret_vertices=False)
    # Map these image plane pixels to the source plane
    beta_pixels = af.img2src(theta_pixels, m, zeta, state)
    # Identify relevant pixels, that map to the source plane region of interest
    rel_pixels = np.nonzero(((beta_offset[0] - beta_pixels[:, 0]) ** 2 + (beta_offset[1] - beta_pixels[:, 1]) ** 2)
                            < beta_lim ** 2)[0]
    theta_pixels_out, beta_pixels_out = [], []
    for pixel_num in rel_pixels:
        theta_pixels_out.append(theta_pixels[pixel_num])
        beta_pixels_out.append(beta_pixels[pixel_num])
    return theta_pixels_out, beta_pixels_out

