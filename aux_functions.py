import numpy as np


def draw_from_dis(dis_x, min_x, max_x, max_y, num):
    """
    A simple function to draw events according to a given 1D distribution.
    :param dis_x: a functional (lambda expression) form of the 1D distribution
    :param min_x: the lower boundary to draw from
    :param max_x: the upper boundary to draw from
    :param max_y: the maximal value of the distribution function in the range {min_x,max_x}
    :param num: the number of output events
    :return: x_out, a numpy array of size (num) with the drawn events
    """
    if num == 0:
        return []
    flag = 0
    x_out = []
    while flag == 0:
        x = np.random.uniform(min_x, max_x, size=2 * num)
        y = np.random.uniform(0, max_y, size=2 * num)
        f = dis_x(x)
        x_out = np.append(x_out, x[y < f])  # check if randomized points are drawn from distribution
        if x_out.shape[0] > num:  # check if x contains values to prevent null vector
            flag = 1
    return x_out[:num]


def img2src(theta, m, zeta, state, mu_flag=False):
    """
    Based on the number l angular positions theta[horizontal,vertical] of an image,
    calculate the angular positions beta[h,v] of the source.
    All angular positions are in units of typical einstein radius given from 'state'
    :param theta: angular position of l images in the lens plane, size (l,2)
    :param m: vector of masses of points sources in units of solar mass
    :param zeta: array of angular position of point sources (horizontal,vertical) (n,2)
    :param state: a dictionary containing all variables of the lens and user definitions
    :param mu_flag: whether or not to analytically calculate the magnification value
    :return: angular position beta[h,v] of source (in units of einstein radii), size (l,2), and magnitude,
    mu size (l,1) of each image theta
    """
    # first, the normalization constant
    G_til = (1 / state['D']) * state['fourGc2'] * (state['rad2sec'] / state['theta_ein']) ** 2
    kappa_bg = state['kappa_bg']
    gamma = state['gamma']
    flat_spec = state['flat_spec']
    m = m.reshape((m.shape[0], 1))
    n = m.shape[0]
    l = theta.shape[0]
    theta_h = theta[:, 0].reshape((l, 1))
    theta_v = theta[:, 1].reshape((l, 1))  # size(l,1)
    zeta_h = zeta[:, 0].reshape((1, n))  # size(1,n)
    zeta_v = zeta[:, 1].reshape((1, n))  # size(1,n)
    d_h = theta_h - zeta_h  # size(l,n)
    d_v = theta_v - zeta_v  # size(l,n)
    d_h2 = d_h ** 2
    d_v2 = d_v ** 2
    d2 = d_h2 + d_v2
    if flat_spec:
        alpha_h = m[0] * np.sum(d_h / d2, axis=1)  # size(l,1)
        alpha_v = m[0] * np.sum(d_v / d2, axis=1)  # size(l,1)
    else:
        alpha_h = (d_h / d2) @ m  # size(l,1)
        alpha_v = (d_v / d2) @ m  # size(l,1)
    alpha = np.hstack((alpha_h.reshape((l, 1)), alpha_v.reshape((l, 1)))) * G_til  # size(l,2)
    beta = theta * np.array([1 - kappa_bg + gamma, 1 - kappa_bg - gamma]) - alpha
    if mu_flag:
        d22 = d2 ** 2
        nu_hh_per_m = ((d_v2 - d_h2) / d22)
        nu_hv_per_m = -2 * (d_h * d_v / d22)
        if flat_spec:
            nu_hh = np.sum(nu_hh_per_m, axis=1) * m[0]  # size(l,1)
            nu_hv = np.sum(nu_hv_per_m, axis=1) * m[0]  # size(l,1)
        else:
            nu_hh = nu_hh_per_m @ m  # size(l,1)
            nu_hv = nu_hv_per_m @ m  # size(l,1)
        A = ((1 - kappa_bg) ** 2 - gamma ** 2 - 2 * G_til * nu_hh * gamma - (G_til ** 2) * (nu_hv ** 2 + nu_hh ** 2))
        A = A.reshape((l, 1))
        if flat_spec:
            dh_nu_hh = -2 * np.sum(d_h * (1 / d22 + 2 * nu_hh_per_m / d2), axis=1) * m[0]
            dv_nu_hh = 2 * np.sum(d_v * (1 / d22 - 2 * nu_hh_per_m / d2), axis=1) * m[0]
            dh_nu_hv = -2 * np.sum(d_v / d2 * nu_hh_per_m, axis=1) * m[0]
            dv_nu_hv = 2 * np.sum(d_h / d2 * nu_hh_per_m, axis=1) * m[0]
        else:
            dh_nu_hh = -2 * (d_h * (1 / d22 + 2 * nu_hh_per_m / d2)) @ m
            dv_nu_hh = 2 * (d_v * (1 / d22 - 2 * nu_hh_per_m / d2)) @ m
            dh_nu_hv = -2 * (d_v / d2 * nu_hh_per_m) @ m
            dv_nu_hv = 2 * (d_h / d2 * nu_hh_per_m) @ m
        dh_A = -2 * G_til * gamma * dh_nu_hh - 2 * G_til ** 2 * (dh_nu_hh * nu_hh + dh_nu_hv * nu_hv)
        dv_A = -2 * G_til * gamma * dv_nu_hh - 2 * G_til ** 2 * (dv_nu_hh * nu_hh + dv_nu_hv * nu_hv)
        return beta, np.hstack((A, dh_A.reshape(l, 1), dv_A.reshape(l, 1)))
    return beta


def lens_gen(zeta_lim, state):
    """
    This function generates a population of point masses with angular positions zeta(horizontal,vertical) according
    to a power law given by spectrum P(m)\propto m^(-mass_spectrum)dm
    All angular positions are in units of arc-sec, and are spread over a circular area of radius zeta_lim
    :param zeta_lim: the angular limit, in arc-sec
    :param state: a dictionary containing all variables of the lens and user definitions
    :return: a list of solar masses m, a list of horizontal position zeta_h and vertical positions zeta_v
    """

    # First, importing all variables from the dictionary 'state'

    m0 = 10 ** state['low_log_mass']
    mf = 10 ** state['high_log_mass']
    mass_spec = state['mass_spectrum']
    fourGc2 = state['fourGc2']
    D = state['D']
    rad2sec = state['rad2sec']
    kappa = state['kappa_mass']
    max_y = m0 ** (-mass_spec)
    if mass_spec == 0:
        mean_m = m0
    else:
        mean_m = (m0 ** 2 * mf ** mass_spec - m0 ** mass_spec * mf ** 2) \
                 / (m0 * mf ** mass_spec - m0 ** mass_spec * mf) * (mass_spec - 1) / (mass_spec - 2)
    Sigma = (1 / (fourGc2 * np.pi)) * D  # The surface mass density (solar-mass per rad^2)
    effective_area = zeta_lim ** 2 * np.pi
    m_tot = kappa * Sigma * effective_area / rad2sec ** 2
    n = np.int(np.round(m_tot / mean_m))
    m = draw_from_dis(lambda x: x ** (-mass_spec), m0, mf, max_y, n)
    zeta_phi = np.random.rand(n) * 2 * np.pi
    zeta_r = np.sqrt(np.random.rand(n) * zeta_lim ** 2)
    zeta_h = np.cos(zeta_phi) * zeta_r
    zeta_v = np.sin(zeta_phi) * zeta_r
    actual_kappa = (np.sum(m) * rad2sec ** 2 / effective_area) / Sigma
    return m, np.vstack((zeta_h, zeta_v)).T, actual_kappa


def mag_binning(beta, mu, boundary, beta_res, beta_offset=[0,0], print_flag=False):
    """

    :param beta:
    :param mu:
    :param boundary:
    :param beta_res:
    :param beta_offset:
    :param print_flag:
    :return:
    """
    max_h = boundary / 2 + beta_offset[0]
    max_v = boundary / 2 + beta_offset[1]
    min_h = -boundary / 2 + beta_offset[0]
    min_v = -boundary / 2 + beta_offset[1]
    n_h = beta_res
    n_v = beta_res
    mu_tot = np.zeros((n_h, n_v))
    if print_flag:
        print('Using approximately ' + str(n_h * n_v * 4 / 1000000) + ' MBytes of memory')
    if n_h * n_v * 4 > 8 * 10 ** 9:  # if file size is larger than 4GB
        return 0
    grid_vec_h, h_step = np.linspace(min_h, max_h, n_h, endpoint=False, retstep=True)
    grid_vec_v, v_step = np.linspace(min_v, max_v, n_v, endpoint=False, retstep=True)
    if print_flag:
        print('Actual horizontal resolution=' + str(np.format_float_scientific(h_step, 2)) + ' arc sec')
        print('Actual vertical resolution=' + str(np.format_float_scientific(v_step, 2)) + ' arc sec')
    beta_grid_h, beta_grid_v = np.meshgrid(grid_vec_h, grid_vec_v)
    if beta.shape[0] > 0:  # if beta list is not empty
        beta_h = beta[:, 0].reshape((beta.shape[0], 1))
        beta_v = beta[:, 1].reshape((beta.shape[0], 1))  # size(l,1)
        mu_h_idx = np.squeeze(np.floor_divide(beta_h - min_h, h_step).astype(int))
        mu_v_idx = np.squeeze(np.floor_divide(beta_v - min_v, v_step).astype(int))
        for i in range(beta.shape[0]):
            if n_h > mu_h_idx[i] >= 0 <= mu_v_idx[i] < n_v:
                mu_tot[mu_h_idx[i], mu_v_idx[i]] += mu
    return beta_grid_h, beta_grid_v, mu_tot.T / (h_step * v_step)


def tessellate_area(boundary, offset, num_of_tiles, ret_vertices=True, ret_centers=True):
    """
    Currently returning the coordinates of the vertices of a square-centered lattice with num_of_tiles tiles.
    Each polygon tile has 4 vertices (only 1 of them are unique!) + 1 center point
    Order is as follows:
    First, the vertices of the polygons, from bottom left to top right, horizontal then vertical
    [(sqrt(num_of_tiles)+1)^2,2].
    Then, an array of the centers of each polygon, from bottom left to top right, horizontal then vertical.
    [num_of_tiles,2]

    :param boundary: a list containing the horizontal, and vertical (half) boundaries.
    :param num_of_tiles: The number of tiles to divide the given area
    :param offset: The center point of the region to tile [h,v]
    :param ret_vertices:
    :param ret_centers:
    :return:
    """
    h_lim = boundary[0]  # horizontal half limit
    v_lim = boundary[1]  # vertical half limit
    num_of_tiles_perrow = int(np.sqrt(num_of_tiles))  # the number of tiles per row
    if ret_vertices:
        h, step_h = np.linspace(-h_lim, h_lim, num_of_tiles_perrow + 1, endpoint=True, retstep=True)\
                          + offset[0]
        v, step_v = np.linspace(-v_lim, v_lim, num_of_tiles_perrow + 1, endpoint=True, retstep=True)\
                          + offset[1]
        grid_h, grid_v = np.meshgrid(h, v)
        true_num_tiles = grid_h.shape[0] * grid_h.shape[1]
        vertices = np.vstack((grid_h.reshape((1, true_num_tiles)), grid_v.reshape((1, true_num_tiles)))).T
    if ret_centers:
        step_h = h_lim * 2 / (num_of_tiles_perrow + 1)
        step_v = v_lim * 2 / (num_of_tiles_perrow + 1)
        h = np.linspace(-h_lim + step_h / 2, h_lim - step_h / 2, num_of_tiles_perrow, endpoint=True) + offset[0]
        v = np.linspace(-v_lim + step_v / 2, v_lim - step_v / 2, num_of_tiles_perrow, endpoint=True) + offset[1]
        grid_h, grid_v = np.meshgrid(h, v)
        true_num_tiles = grid_h.shape[0] * grid_h.shape[1]
        centers = np.vstack((grid_h.reshape((1, true_num_tiles)), grid_v.reshape((1, true_num_tiles)))).T
        if ret_vertices:
            return vertices, centers
        else:
            return centers
    else:
        return vertices

