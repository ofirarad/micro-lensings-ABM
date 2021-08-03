import numpy as np
import matplotlib as mpl
import time
import os
import ipm
import irs
import abm
import aux_functions as af
import casutic_light_curve as clc

mpl.use('tkagg')  # To solve matplotlib problem with mac OS
# ----------------------------------------------------------------------------------------------------------------------
#                                              USER INPUT START
# ----------------------------------------------------------------------------------------------------------------------
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
source_size = 7 * 10 ** 13  # the desired point source radius in cm (half width)

mass_spectrum = 2.63  # spectrum of the power-law mass distribution, of point mass lenses
low_log_mass = -1  # Low boundary for the mass of lenses in log10(solar mass)
high_log_mass = np.log10(1.5)  # High boundary for the mass of lenses in log10(solar mass)
Mtypical = 1  # The typical mass for the relevant Einstein radius in solar mass - only for presentation purposes

# User definitions- numerical
home_dir = os.getcwd()  # main directory of the project; Can be manually set if desired
save_flag = True  # whether to save the list of sources and magnification to a numpy dump file, and to save the figures.
plot_flag = True  # whether to show plots (caustic maps and light curves) while running
plot_arcsec = True  # Whether to use arc sec for the plot units.
epsilon = np.sqrt(2.1)  # The size increment factor of FOV in the lens plane (after considering the magnification)
zeta_mar = 1.2  # Range of point masses locations in the lens plane in units of epsilon
vt = 1000  # transverse velocity in km/s in the image plane
light_curve_yr = 5  # length of transverse motion in years, the source plane FOV
method = 'IRS'  # The chosen method for creating the caustics. Can be either 'IPM', 'IRS' or 'ABM'
user_seed = 276190  # seed value for replicating random distributions

beta_res = 60  # the resolution in pixels per axis for plotting the source FOV (only for IRS and IPM)
beta_zoom = 5  # The zoom-in factor for plotting the source FOV
max_memory = 0.01  # maximum memory usage for arrays in intermediate calculations, in GB

# IRS settings
rays_per_pixel = 1  # rays per non-lensed image plane pixels, for IRS method

# IPM settings
num_of_tiles = 0  # the number of tiles to tessellate the image plane; a square integer. 0 for automatic value.
IPM_l_parameter = 10  # the factor by which the IPM tiles are smaller than the lowest-mass Einstein-radius
IRS_nlin_tiles = 16 ** 2  # 16 ** 2  # the number of rays to split the non-linear tiles to; must be a square integer

# AOP settings
n_pixels = 10  # number of image plane pixels per dimension in each division; could change to increase efficiency
eta_ratio = 0.8
n_steps = 25  # The number of initial light curve time steps. Half the number of final time steps
beta_0 = 2  # the factor of initial search boundary in the source plane, in units of FOV.
kernel_flag = True  # Whether to bin rays in the source plane and apply a Gaussian profile before creating light curve
# ----------------------------------------------------------------------------------------------------------------------
#                                                       END OF USER INPUT
# ----------------------------------------------------------------------------------------------------------------------
# The dictionary "parameters" stores the current state of the simulation. Including all user input and implied variables
# ----------------------------------------------------------------------------------------------------------------------
parameters = {'fourGc2': fourGc2, 'rad2sec': rad2sec, 'kappa_frac': kappa_frac, 'mu_initial': mu_initial, 'DS': DS,
              'DL': DL,
              'DLS': DLS, 'home_dir': home_dir, 'save_flag': save_flag, 'plot_flag': plot_flag,
              'plot_arcsec': plot_arcsec, 'epsilon': epsilon, 'zeta_mar': zeta_mar, 'rays_per_pixel': rays_per_pixel,
              'num_of_tiles': num_of_tiles, 'max_memory': max_memory,
              'IPM_l_parameter': IPM_l_parameter, 'IRS_nlin_tiles': IRS_nlin_tiles, 'source_size': source_size,
              'eta_ratio': eta_ratio,
              'n_pixels': n_pixels, 'n_steps': n_steps, 'beta_0': beta_0, 'vt': vt, 'light_curve_yr': light_curve_yr,
              'kernel_flag': kernel_flag, 'beta_res': beta_res, 'method': method, 'beta_zoom': beta_zoom,
              'low_log_mass': low_log_mass, 'high_log_mass': high_log_mass, 'Mtypical': Mtypical,
              'mass_spectrum': mass_spectrum}
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
    flat_spec = True
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
    beta_grid_h, beta_grid_v, mu_grid, nlin_tiles = ipm.ipm_method(parameters)

if method == 'IRS':
    beta_grid_h, beta_grid_v, mu_grid = irs.irs_method(parameters)

if method == 'ABM':
    time_steps, light_curve, time_steps_perp, light_curve_perp = abm.abm_method(parameters)

# performing light curve analysis for caustic maps
if method == 'IRS' or method == 'IPM':
    pixel2cm = (DS * beta_boundary * theta_ein / rad2sec * pc2cm) / beta_res  # size in cm of source plane pixel
    # half width of Gaussian luminosity distribution for point source, in pixels. We define the source size as 3 sigma.
    half_width = int(source_size / 3 / pixel2cm)
    time_steps, light_curve = clc.light_curve(half_width, beta_res, beta_boundary, mu_grid, vt_ein, shear_parallel=True)
    time_steps_perp, light_curve_perp = clc.light_curve(half_width, beta_res, beta_boundary, mu_grid, vt_ein,
                                        shear_parallel=False)
# ----------------------------------------------------------------------------------------------------------------------
# Saving and plotting
# ----------------------------------------------------------------------------------------------------------------------
current_date = time.strftime("%Y%m%d_%H%M", time.localtime())
if save_flag:
    if method == 'ABM':
        print('.........Saving the light curves data as dump file')
        array2save = np.array([time_steps, light_curve, time_steps_perp, light_curve_perp, parameters], dtype=object)
        np.save(home_dir + '/data_out/' + current_date, array2save)
        print('.........Saving figure')
    elif method == 'IRS' or method == 'IPM':
        print('.........Saving the sources locations and magnifications as dump file')
        array2save = np.array([beta_grid_h, beta_grid_v, mu_grid, parameters], dtype=object)
        np.save(home_dir + '/data_out/' + current_date +'caustic', array2save)
        print('.........Saving the light curves data as dump file')
        array2save = np.array([time_steps, light_curve, time_steps_perp, light_curve_perp, parameters], dtype=object)
        np.save(home_dir + '/data_out/' + current_date, array2save)
        print('.........Saving figure')

if method == 'ABM':
    af.light_curve_plot(time_steps, light_curve, time_steps_perp, light_curve_perp, current_date, parameters)
elif method == 'IRS' or method == 'IPM':
    af.caustic_plot(beta_grid_h, beta_grid_v, mu_grid, current_date, parameters)
    af.light_curve_plot(time_steps, light_curve, time_steps_perp, light_curve_perp, current_date, parameters)


print('Totally finished in ' + str(time.time() - mega_start_time))
