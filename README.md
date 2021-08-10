10-Aug-2021

Ofir Arad
# Gravitational Micro-lensing simulation - Adaptive Boundary Method
This project presents a novel method for simulating the light curves of stellar sources passing through the caustics of a gravitational lens.
After setting the physical parameters of the lens, the code simulates a distribution of masses that complies with the chosen parameters. Then, the code generates two light curves (parallel and perpendicular to the lens shear), by using one of three methods:
1) *ABM* - Adaptive Boundary Method (Zitrin et. al)
2) *IRS* - Inverse Ray Shooting
3) *IPM* - Inverse Polygon Method (Mediavilla 2006)
## General details
The code is based on python, and specifically numpy for its numerous computations. The pipeline is as follows:
Read user parameters --> derive additional parameters --> generate gravitational lens --> generate light curve --> save and plot.
To run the code, simply set all input variables as desired, and run the main.py file.
Before running, make sure the following python libraries are installed:
*numpy, scipy, matplotlib, time, os, joblib, multiprocessing*.

All the parameters, and properties of the gravitational lens are stored in a python dictionary called *parameters*, so that any additional function can access the state of the gravitational lens at any time.
**All** user inputs are set in a dedicated section in the main.py file, no need to access any other section. The output of the code is saved to a unique folder, and contains the light curve data and caustics (if relevant) as csv files, as well as figures as png files.

For all calculations, angular coordinates are in units of Einstein radius of the typical mass (set as input). For ease of use, we denote all source plane coordinates with beta, all image plane coordinates with theta, and the lens-masses coordinates with zeta. 
This is reflected in the appropriate variable names.

## Generating the gravitational lens
The first stage, common among all three methods above concerns the distribution of point masses that comprise the lens.
The required input parameters are the cosmological angular-diameter distances, the desired convergence of the lens as well as the fraction of it due to point masses, and the desired magnification of the lens.
Given these, the shear of the lens will be automatically calculated. The point masses in the lens are drawn according to a power-law distribution of which spectrum is given as an input parameter. The mass interval to draw from is deduced from user input as well.

The spatial region in which masses are drawn is determined by the source plane field of view (FOV) in the following manner. First, the source plane FOV is derived from the source transverse velocity and the chosen duration of the transverse motion.
Then, we define the region of interest in the image plane as a rectangle with sides *FOV* x *mu* x *epsilon*, where *mu* is the tangential or radial magnification scale (determined by the shear and convergence), and *epsilon* is a margin parameter set by the user.
Finally, the mass distribution region is a circle of which radius is proportional to the radius of the circumscribed circle of the image plane region of interest. The proportionality factor is given as the user input *zeta_mar*.  

The masses of the lens are drawn randomly, depending on a random seed variable *user_seed*. Two code runs with the same seed and same parameters, will produce the same lens.

## Creating the light curves
For each method, a set of parameters exists for the user to set. These are described in the code, as well as in the corresponding literatures.
By setting the variable *method* the user can choose which method to use.
While IRS or IPM first generate a caustic map, and then apply the transverse motion, ABM does not produce the caustic map. As a result, it is much faster for higher resolutions (approaching scales of solar radius), but inefficient if one wishes to examine a large amount of light curves per lens setting.

The IRS method is the only one of the three methods that is parallel-computed. This is done using the running system CPU.
