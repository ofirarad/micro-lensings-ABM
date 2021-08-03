import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import time
import aux_functions as af


def irs_method(state):
    """
    The main routine for running the gravitational lensing simulation using the Inverse Ray Shooting method
    :param state: a dictionary containing all variables of the lens and user definitions
    :return: beta_grid_h, beta_grid_v, mu_grid, beta, theta
    """

    # First, importing all variables from the dictionary 'state'
    theta_ein2cm = state['theta_ein2cm']
    beta_boundary = state['beta_boundary']
    beta_res = state['beta_res']
    epsilon = state['epsilon']
    mu_h = state['mu_h']
    mu_v = state['mu_v']
    m = state['m']
    zeta = state['zeta']
    max_memory = state['max_memory']
    rays_per_pixel = state['rays_per_pixel']

    pixel2cm = theta_ein2cm * beta_boundary / beta_res  # size of 1 pixel in cm in the source plane
    print('The physical size of 1 pixel is ' + str(beta_boundary / beta_res) + ' Einstein radii\nor ' + str(
        np.format_float_scientific(pixel2cm, 2)) + ' cm in the source plane\n')

    theta_boundaries = [epsilon * mu_h * beta_boundary / 2,
                        epsilon * mu_v * beta_boundary / 2]
    # The number of images to draw in IRS method, assuming an ellipse in the image plane
    num_of_img = int((beta_res * epsilon) ** 2 * mu_v * mu_h * rays_per_pixel)
    print('A total of ' + str(num_of_img) + ' images for IRS method')
    state['num_of_img'] = num_of_img
    print(str(num_of_img / beta_res ** 2) + ' rays per source plane pixels')
    # The area in (Einstein-radii)^2 that each ray uniquely occupies
    s_ray = (epsilon ** 2 * mu_h * mu_v * beta_boundary ** 2) / num_of_img

    l_tmp = int(max_memory / m.shape[0] * 10 ** 9 / 8)  # the maximum number of images to vector-compute
    n_runs = max(int(num_of_img / l_tmp), 1)  # the number of sub arrays to vector-compute
    print('Max memory for array: ' + str(l_tmp * m.shape[0] * 8 / 10 ** 9) + 'GB')
    mu_grid = np.zeros((beta_res, beta_res))  # this will save the total number of rays per cell in the source plane
    start_time = time.time()
    theta = []
    beta = []
    num_cores = multiprocessing.cpu_count()
    print(str(num_cores) + ' active CPU cores')
    # starting the parallel routine, the variable mu_grid_temp_array is just a placeholder.
    mu_grid_temp_array = Parallel(n_jobs=num_cores, require='sharedmem')\
        (delayed(parallel_irs)(i,mu_grid,l_tmp,n_runs,s_ray,theta_boundaries,start_time,state) for i in range(n_runs))

    if n_runs * l_tmp < num_of_img:  # if some values are left
        # Drawing images locations
        theta = random_image_draw(int(num_of_img - n_runs * l_tmp), theta_boundaries[0], theta_boundaries[1])
        # Calculating locations of sources and corresponding magnitudes
        beta = af.img2src(theta, m, zeta, state)
        # Binning sources magnification
        beta_grid_h, beta_grid_v, mu_grid_temp = af.mag_binning(beta, s_ray, beta_boundary, beta_res)
        mu_grid += mu_grid_temp
        print('Finished shooting in ' + str(time.time() - start_time) + 's')
    else:
        print('Finished shooting in ' + str(time.time() - start_time) + 's')
        beta = np.ones(2, 2)  # Just so that the next line can run smoothly and return beta_grid_h and beta_grid_v
        beta_grid_h, beta_grid_v, mu_grid_temp = af.mag_binning(beta, s_ray, beta_boundary, beta_res)

    return beta_grid_h, beta_grid_v, mu_grid


def parallel_irs(i,mu_grid,l_tmp,max_runs,s_ray,theta_boundaries,start_time,state):
    """
    Auxiliary function for the parallel computation of the ray shooting simulation, within the scope of irs_method
    :param i: an index for order of parallelization
    :param mu_grid: The magnification grid
    :param l_tmp: The number of rays to work on withing the same batch
    :param max_runs: The maximum runs of this parallel method
    :param s_ray: The image plane area each ray individually takes
    :param theta_boundaries: The half boundaries in the image plane, horizontal and vertical
    :param start_time: The initial start time, as a time object (time library)
    :param state: The dictionary containing all parameters of the lens, and all user inputs
    :return: dummy variable, results are saved under the outer-scope variable mu_grid
    """
    beta_boundary = state['beta_boundary']
    beta_res = state['beta_res']
    m = state['m']
    zeta = state['zeta']
    # Drawing images locations theta
    theta = random_image_draw(l_tmp, theta_boundaries[0], theta_boundaries[1])
    # Calculating locations of sources and corresponding magnitudes
    beta = af.img2src(theta, m, zeta, state)
    # Binning sources magnification
    beta_grid_h, beta_grid_v, mu_grid_temp = af.mag_binning(beta, s_ray, beta_boundary, beta_res)
    mu_grid += mu_grid_temp
    temp_t = time.time() - start_time
    if i % (max(1, int(max_runs / 4000))) == 0:
        print('Finished ' + str(round((i + 1) * 100 / max_runs, 5)) + '% in ' + str(round(temp_t)) +
              's; ~' + str(round(temp_t * (max_runs / (i + 1) - 1))) + 's remaining')
    return 0


def random_image_draw(num_of_img, h_lim, v_lim,ellipse=False):
    """
    This functions randomly draws images in a given rectangle, or if specified, in the enclosed ellipse.
    :param num_of_img: number of images to draw
    :param h_lim: half horizontal boundary
    :param v_lim: half vertical boundary
    :param ellipse: whether to draw from an ellipse of semi-major and semi-minor axes h_lim and v_lim
    :return: numpy array of size (num_of_img,2), of horizontal and vertical coordinates
    """
    if ellipse:
        ellipse_box_num = int(
            num_of_img * 1.27324)  #  to get ~ num_of_img images in the contained ellipse
        theta = np.hstack((h_lim * (2 * np.random.rand(ellipse_box_num).reshape(ellipse_box_num, 1) - 1),
                           v_lim * (2 * np.random.rand(ellipse_box_num).reshape(ellipse_box_num, 1) - 1)))
        theta = theta[(theta[:, 0] / h_lim) ** 2 + (theta[:, 1] / v_lim) ** 2 < 1]
    else:
        theta = np.hstack((h_lim * (2 * np.random.rand(num_of_img).reshape(num_of_img, 1) - 1),
                           v_lim * (2 * np.random.rand(num_of_img).reshape(num_of_img, 1) - 1)))
    return theta


