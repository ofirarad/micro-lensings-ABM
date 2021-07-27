import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from scipy.signal import find_peaks
import ipm
import irs
import abm
import aux_functions as af

mpl.use('tkagg') #  To solve matplotlib problem with mac OS


def plot_routine(zeta, beta, beta_grid_h, beta_grid_v, mu_tot, theta, parameters):
    """
    :param beta_grid_h:
    :param beta_grid_v:
    :param mu_tot:
    :param theta:
    :param save_flag:
    :return:
    """
    kappa = parameters['kappa']
    gamma = np.round(parameters['gamma'],3)
    mu_initial = parameters['mu_initial']
    kappa_frac = parameters['kappa_frac']
    D = parameters['D']
    DS = parameters['DS']
    method = parameters['method']
    plot_arcsec = parameters['plot_arcsec']
    if method == 'IPM':
        num_of_img = parameters['num_of_tiles']
    else:
        num_of_img = parameters['num_of_img']
        rays_per_pixel = parameters['rays_per_pixel']
    zoom = parameters['beta_zoom']
    save_flag = parameters['save_flag']
    home_dir = parameters['home_dir']
    theta_ein = parameters['theta_ein']

    if plot_arcsec:
        zeta *= theta_ein * 10 ** 6
        theta *= theta_ein * 10 ** 6
        beta *= theta_ein * 10 ** 6
        beta_grid_h *= theta_ein * 10 ** 6
        beta_grid_v *= theta_ein * 10 ** 6
        plot_units = 'Angular position ($\mu$")'
    else:
        plot_units = 'Angular position ($\u03BE_0$)'

    zeta_lim = max(np.max(zeta[:, 0]), np.max(zeta[:, 1]))
    zeta_lim_h = zeta_lim
    zeta_lim_v = zeta_lim
    theta_lim = max(np.max(theta[:, 0]), np.max(theta[:, 1]))
    theta_lim_h = theta_lim
    theta_lim_v = theta_lim
    beta_lim = -beta_grid_h[0, 0]

    # Creating figures
    print('.........Creating figures.........')
    fig, ax = plt.subplots(2, 2, constrained_layout=True, figsize=(8, 9))

    zoom_h_min = int(beta_grid_h.shape[0] / 2 * (1 - 1 / zoom))
    zoom_v_min = int(beta_grid_h.shape[1] / 2 * (1 - 1 / zoom))
    zoom_h_max = zoom_h_min + int(beta_grid_h.shape[0] / zoom)
    zoom_v_max = zoom_v_min + int(beta_grid_h.shape[1] / zoom)

    start_time = time.time()
    print('.........Distribution of lenses')
    zoom_box_0 = mpl.patches.Circle((0, 0), radius=beta_lim, ec="cyan", fill=False)
    ax[0, 0].add_patch(zoom_box_0)
    ax[0, 0].scatter(zeta[:, 0], zeta[:, 1], s=11)
    ax[0, 0].set_title('Distribution of lenses')
    ax[0, 0].set_xlabel(plot_units)
    ax[0, 0].set_ylabel(plot_units)
    ax[0, 0].set_aspect('equal')
    ax[0, 0].set_xlim(-zeta_lim_h, zeta_lim_h)
    ax[0, 0].set_ylim(-zeta_lim_v, zeta_lim_v)
    # ax[0,0].set_facecolor('black')
    print('Finished in ' + str(time.time() - start_time) + 's')

    ax[1, 0].set_xlim(-theta_lim_h, theta_lim_h)
    ax[1, 0].set_ylim(-theta_lim_v, theta_lim_v)
    sampled_idx = np.random.randint(0, theta.shape[0] - 1, 1000, dtype=int)
    ax[1, 0].scatter(theta[sampled_idx, 0], theta[sampled_idx, 1], c='green', s=11)
    ax[1, 0].scatter(beta[sampled_idx, 0], beta[sampled_idx, 1], c='magenta', s=10, alpha=0.1)
    ax[1, 0].set_title('Images (green) and sources (magenta)')
    ax[1, 0].set_xlabel(plot_units)
    ax[1, 0].set_ylabel(plot_units)
    ax[1, 0].set_aspect('equal')
    zoom_box_1 = mpl.patches.Circle((0, 0), radius=beta_lim, ec="cyan", fill=False)
    ax[1, 0].add_patch(zoom_box_1)

    ax[0, 1].set_title('Source plane; ' + str(beta_grid_h.shape[0]) + 'x' + str(beta_grid_h.shape[0]) + ' pixel')
    ax[0, 1].set_xlabel(plot_units)
    ax[0, 1].set_ylabel(plot_units)
    ax[0, 1].set_xlim(-beta_lim, beta_lim)
    ax[0, 1].set_ylim(-beta_lim, beta_lim)
    ax[0, 1].set_aspect('equal')
    zoom_box = mpl.patches.Rectangle((beta_grid_h[0, zoom_h_min], beta_grid_v[zoom_v_min, 0]),
                                     (beta_grid_h[0, -1] - beta_grid_h[0, 0]) / zoom,
                                     (beta_grid_v[-1, 0] - beta_grid_v[0, 0]) / zoom, ec="red", fill=False, lw=1,
                                     alpha=0.5)
    radius_interest = mpl.patches.Circle((0, 0), radius=beta_lim, ec="cyan", fill=False)
    ax[0, 1].add_patch(zoom_box)
    ax[0, 1].add_patch(radius_interest)
    start_time = time.time()
    print('.........Magnification of sources')
    cs3 = ax[0, 1].pcolormesh(beta_grid_h, beta_grid_v, mu_tot, cmap='cubehelix', norm=mpl.colors.LogNorm())
    plt.colorbar(cs3, ax=ax[0, 1])
    ax[0, 1].set_facecolor('black')
    print('Finished in ' + str(time.time() - start_time) + 's')

    ax[1, 1].set_title('Source plane; ' + str(zoom) + 'x zoom')
    ax[1, 1].set_xlabel(plot_units)
    ax[1, 1].set_ylabel(plot_units)
    ax[1, 1].set_aspect('equal')
    ax[1, 1].set_xlim(-beta_lim / zoom, beta_lim / zoom)
    ax[1, 1].set_ylim(-beta_lim / zoom, beta_lim / zoom)
    ax[1, 1].set_facecolor('black')
    ax[1, 1].spines['left'].set_color('red')
    ax[1, 1].spines['right'].set_color('red')
    ax[1, 1].spines['top'].set_color('red')
    ax[1, 1].spines['bottom'].set_color('red')
    cs4 = ax[1, 1].pcolormesh(beta_grid_h[zoom_h_min:zoom_h_max, zoom_v_min:zoom_v_max],
                              beta_grid_v[zoom_h_min:zoom_h_max, zoom_v_min:zoom_v_max]
                              , mu_tot[zoom_h_min:zoom_h_max, zoom_v_min:zoom_v_max], cmap='cubehelix',
                              norm=mpl.colors.LogNorm())
    plt.colorbar(cs4, ax=ax[1, 1])

    current_date = time.strftime("%Y%m%d_%H%M", time.localtime())
    if method == 'IRS' or method == 'AOP':
        chart_title = current_date + '\n$\u03BE_0$=' + str(
            np.format_float_scientific(theta_ein, precision=2)) + '", and ' \
                      + str(
            np.format_float_scientific(np.sqrt(fourGc2 / D) * DS, 2)) + 'pc in the source plane; $\kappa$=' + str(
            np.round(kappa, 2)) \
                      + ' $\gamma$=' + str(gamma) + '; $\kappa_*/\kappa=$' + str(kappa_frac) + ' $\mu=$' +\
                      str(mu_initial) + '\n' + str(m.shape[0]) + ' point masses; M_tot=' + \
                      str(np.format_float_scientific(np.sum(m), precision=2)) + ' Solar mass; ' \
                      + str(np.format_float_scientific(num_of_img, 2)) + ' rays, with ' + str(
            rays_per_pixel) + ' rays per pixel'
    elif method == 'IPM':
        chart_title = current_date + '\n$\u03BE_0$=' + str(
            np.format_float_scientific(theta_ein, precision=2)) + '", and ' \
                      + str(
            np.format_float_scientific(np.sqrt(fourGc2 / D) * DS, 2)) + 'pc in the source plane; $\kappa$=' + str(
            np.round(kappa, 2)) \
                      + ' $\gamma$=' + str(gamma) + '; ' + str(m.shape[0]) + ' point masses\nM_tot=' + \
                      str(np.format_float_scientific(np.sum(m),
                                                     precision=2)) + ' Solar mass; Inverse polygon method with ' \
                      + str(np.format_float_scientific(num_of_img, 2)) + ' tiles'

    fig.suptitle(chart_title)

    if save_flag:
        start_time = time.time()
        print('.........Saving figure')
        fig.savefig(home_dir + '/test figures/'
                    + current_date + '.png', format="png")
        print('Finished in ' + str(time.time() - start_time) + 's')
    else:
        plt.show()

    return 0


# -------------------------------------------------------------------
# -------------------------------------------------------------------
# User definitions- physical
fourGc2 = 1.91422 * 10 ** (-13)  # the constant 4*G/c^2 in units of pc/solar mass
rad2sec = 3600 * 180 / np.pi  # multiply to transform angular positions from radians to arc seconds
pc2cm = 3.0856775814913673 * 10 ** 18  # pc to cm

kappa = 0.85  # 0  # General convergence
kappa_frac = 0.1  # fraction of the convergence that is in point sources
mu_initial = 250  # total magnification value, will determine the background shear
DS = 1.7616 * 10 ** 9  # z=1.5; The angular diameter distance to source in pc
DL = 9.1183 * 10 ** 8  # z=0.3; The angular diameter distance to lens in pc
DLS = 1.2875 * 10 ** 9  # The angular diameter distance from lens to source in pc
vt = 1000  # transverse velocity in km/s in the image plane

mass_spectrum = 2.63 # spectrum of the power-law mass distribution, of point mass lenses
low_log_mass = -1  # Low boundary for the mass of lenses in log10(solar mass)
high_log_mass = np.log10(1.5)  # High boundary for the mass of lenses in log10(solar mass)
Mtypical = 1  # The typical mass for the relevant Einstein radius in solar mass - only for presentation purposes

# User definitions- numerical
home_dir = '/Users/ofirarad/Documents/Academic stuff/gravitational microlensing'
save_flag = False  # whether to save the list of sources and magnification to a numpy dump file
plot_flag = True  # whether to plot the caustic map
plot_arcsec = True  # Whether to use arc sec for the plot units.
epsilon = np.sqrt(2.1)  # The size increment factor of FOV in the lens plane (after considering the magnification)
zeta_mar = 1.2  # Range of point masses locations in the lens plane in units of epsilon
vt = 1000  # transverse velocity in km/s in the image plane
light_curve_yr = 5  # length of transverse motion in years, the source plane FOV
method = 'ABM'  # The chosen method for creating the caustics. Can be either 'IPM', 'IRS' or 'ABM'
beta_res = 120  # the resolution in pixels per axis for plotting the source FOV (only for IRS and IPM)
beta_zoom = 5  # The zoom-in factor for plotting the source FOV
max_memory = 0.01  # maximum memory usage for arrays in intermediate calculations, in GB
user_seed = 276190  # seed value for replicating random distributions

# IRS settings
rays_per_pixel = 1  # rays per non-lensed image plane pixels, for IRS method

# IPM settings
num_of_tiles = 0  # the number of tiles to tessellate the image plane; a square integer. 0 for automatic value.
IPM_l_parameter = 10  # the factor by which the IPM tiles are smaller than the lowest-mass Einstein-radius
IRS_nlin_tiles = 16 ** 2  # 16 ** 2  # the number of rays to split the non-linear tiles to; must be a square integer

# AOP settings
source_size = 7 * 10 ** 13  # the desired point source size in cm (half width)
n_pixels = 12  # number of image plane pixels per dimension in each division; could change to increase efficiency
eta_ratio = 0.8
n_steps = 25  # The number of initial light curve time steps. Half the number of final time steps
beta_0 = 2  # the factor of initial search boundary in the source plane, in units of FOV.
kernel_flag = True  # Whether to bin rays in the source plane and apply a Gaussian profile before creating light curve
# ----------------------------------------------------------------------------------------------------------------------
#                                                       END OF USER INPUT
# ----------------------------------------------------------------------------------------------------------------------
# The dictionary "parameters" stores the current state of the simulation. Including all user input and implied variables
# ----------------------------------------------------------------------------------------------------------------------
parameters = {'fourGc2':fourGc2,'rad2sec':rad2sec,'kappa_frac':kappa_frac,'mu_initial':mu_initial,'DS':DS,'DL':DL,
              'DLS':DLS,'home_dir':home_dir,'save_flag':save_flag,'plot_arcsec':plot_arcsec,'epsilon':epsilon,
              'zeta_mar':zeta_mar,'rays_per_pixel':rays_per_pixel,'num_of_tiles':num_of_tiles,'max_memory':max_memory,
              'IPM_l_parameter':IPM_l_parameter,'IRS_nlin_tiles':IRS_nlin_tiles,'source_size':source_size,
              'eta_ratio':eta_ratio,'beta_0':beta_0,
              'n_pixels':n_pixels,'n_steps':n_steps,'beta_0':beta_0,'vt':vt,'light_curve_yr':light_curve_yr,
              'kernel_flag':kernel_flag,'beta_res':beta_res,'method':method,'beta_zoom':beta_zoom,
              'low_log_mass':low_log_mass,'high_log_mass':high_log_mass,'Mtypical':Mtypical,
              'mass_spectrum':mass_spectrum}
# ----------------------------------------------------------------------------------------------------------------------
start_time = time.time()
mega_start_time = start_time
# ----------------------------------------------------------------------------------------------------------------------
# First, calculating implied variables
# ----------------------------------------------------------------------------------------------------------------------
kappa_mass = kappa * kappa_frac  # Convergence of point sources
parameters['kappa_mass'] = kappa_mass
kappa_bg = kappa - kappa_mass  # Smooth background convergence
parameters['kappa_bg'] = kappa_bg
# Shear of background; By definition, positive shear will stretch the vertical image axis compared to source plane
gamma = np.sqrt((kappa ** 2 * mu_initial - 2 * kappa * mu_initial + mu_initial - 1) / mu_initial)
parameters['gamma'] = gamma
D = DL * DS / DLS  # The effective angular diameter distance in pc
parameters['D'] = D
print('Effective lensing angular diameter distance: ' + str(np.format_float_scientific(D, 2)) + ' pc')
theta_ein_1 = rad2sec * np.sqrt(fourGc2 / D)  # Einstein radius of 1 solar mass in arcsec
parameters['theta_ein_1'] = theta_ein_1
print('1 solar-mass Einstein radius in the source plane: ' + str(
    np.format_float_scientific(np.sqrt(fourGc2 / D) * DS, 2)) + ' pc')
theta_ein = theta_ein_1 * np.sqrt(Mtypical)  # Einstein radius of typical mass in arcsec
parameters['theta_ein'] = theta_ein
print('Typical point mass Einstein radius in the source plane: ' + str(
    np.format_float_scientific(np.sqrt(Mtypical * fourGc2 / D) * DS, 2)) + ' pc')
print('Einstein radius of lowest mass, ' + str(10 ** low_log_mass) + ' solar-mass: ' + str(
    theta_ein_1 * np.sqrt(10 ** low_log_mass)) + ' arcsec')
print('And in terms of the typical Einstein radius: ' + str(np.sqrt(10 ** low_log_mass / Mtypical)))
theta_ein2cm = (DS * theta_ein / rad2sec * pc2cm)  # size of 1 Einstein radius in cm in the source plane
parameters['theta_ein2cm'] = theta_ein2cm
vt_ein = (vt * 100000 / pc2cm) / (DL * theta_ein / rad2sec)  # The source velocity in units of Einstein radii / s
parameters['vt_ein'] = vt_ein
beta_boundary = vt_ein * light_curve_yr * 31557600  # The relevant source plane field of view in Einstein radii
parameters['beta_boundary'] = beta_boundary
print('The source plane field of view is ' + str(beta_boundary) + ' Einstein radii, and '
      + str(theta_ein2cm * beta_boundary) + 'cm')
# ----------------------------------------------------------------------------------------------------------------------
# Drawing point sources
# ----------------------------------------------------------------------------------------------------------------------
mu_h = 1 / (1 - kappa + gamma)  # (Not the final) radial magnification factor
mu_v = 1 / (1 - kappa - gamma)  # (Not the final) tangential magnification factor
# Radius of area in which point masses are drawn, in arc-sec!
zeta_lim = zeta_mar * epsilon * np.sqrt(mu_v ** 2 + mu_h ** 2) * beta_boundary * theta_ein / 2
np.random.seed(user_seed)  # To keep the same mass distribution
if kappa_mass > 0:
    print('.........Drawing point sources')
    m, zeta, point_source_kappa = af.lens_gen(zeta_lim, parameters)
    if mass_spectrum == 0:
        flat_spec = True
    else:
        flat_spec = False
    print(str(m.shape[0]) + ' point sources')
    print('Finished in ' + str(time.time() - start_time) + 's')
    kappa = kappa_bg + point_source_kappa  # True value of convergence
    print('kappa=' + str(kappa))
else:
    kappa = kappa_bg  # True value of convergence
    m = np.zeros((1, 1))
    zeta = np.zeros((1, 2))
zeta /= theta_ein  # Transforming the lenses positions to units of Einstein radii
parameters['zeta'] = zeta
parameters['m'] = m
parameters['flat_spec'] = flat_spec
# Calculating final horizontal and vertical magnification of FOV
mu_h = 1 / (1 - kappa + gamma)
mu_v = 1 / (1 - kappa - gamma)
parameters['mu_h'] = mu_h
parameters['mu_v'] = mu_v
parameters['kappa'] = kappa
# ----------------------------------------------------------------------------------------------------------------------
# Start of simulation
# ----------------------------------------------------------------------------------------------------------------------

if method == 'IPM':
    beta_grid_h, beta_grid_v, mu_grid, nlin_tiles, beta, theta = ipm.ipm_method(parameters)

if method == 'IRS':
    beta_grid_h, beta_grid_v, mu_grid, beta, theta = irs.irs_method(parameters)

if method == 'ABM':
    time_steps, light_curve, time_steps_perp, light_curve_perp = abm.abm_method(parameters)


    # plotting
    fig,ax=plt.subplots(2,1, figsize=(8,8))
    current_date = time.strftime("%Y%m%d_%H%M", time.localtime())
    chart_title = current_date + '\n$\u03BE_0$=' + str(
        np.format_float_scientific(theta_ein, precision=2)) + '", and ' \
                  + str(
        np.format_float_scientific(np.sqrt(fourGc2 / D) * DS, 2)) + 'pc in the source plane; $\kappa$=' + str(
        np.round(kappa, 2)) + ', $\kappa_*/\kappa=$' + str(kappa_frac) + ' ,$\gamma$=' + str(np.round(gamma,3)) \
                  + '\n $\mu=$' +str(mu_initial) + '; ' + str(m.shape[0]) + ' point masses; M_tot=' + \
                  str(np.format_float_scientific(np.sum(m),precision=2)) + ' Solar mass; Source size= ' + \
                  str(np.format_float_scientific(source_size)) + 'cm \n Adi-Ofir path method with ' \
                  + str(int(np.floor(parameters['num_iter']))) + '+1 iterations and ' + str(n_pixels) + \
                  ' image divisions per iteration\n'
    fig.suptitle(chart_title)
    fig.subplots_adjust(top=0.85)
    ax[0].plot(time_steps, np.log10(light_curve+1), 'b')
    ax[1].plot(time_steps_perp, np.log10(light_curve_perp+1), 'r')

    ax[1].set_xlabel('Time (yr)')
    ax[1].set_ylabel('$\log_{10}(\mu)$')
    ax[0].set_ylabel('$\log_{10}(\mu)$')

# ----------------------------------------------------------------------------------------------------------------------
# Saving and plotting
# ----------------------------------------------------------------------------------------------------------------------
current_date = time.strftime("%Y%m%d_%H%M", time.localtime())
if save_flag:
    if method=='ABM':
        print('.........Saving the light curves data as dump file')
        np.save(home_dir + '/AOP_trials/' + current_date, [time_steps, light_curve,time_steps_perp, light_curve_perp,
                                                           parameters])
        print('.........Saving figure')
        fig.savefig(home_dir + '/AOP_trials/'+ current_date + '.png', format="png")
    elif method=='IRS' or method=='IPM':
        print('.........Saving the sources locations and magnifications as dump file')
        print('Approximate file size: ' + str(beta.shape[0] * 3 * 8 / 10 ** 6) + 'MB')
        np.save(home_dir + '/test data/' + current_date, [beta_grid_h, beta_grid_v, mu_grid, parameters])
elif method=='ABM':
    plt.show()

if parameters['plot_flag']:
    plot_routine(zeta, beta, beta_grid_h, beta_grid_v, mu_grid, theta, parameters=parameters)

print('Totally finished in ' + str(time.time() - mega_start_time))
