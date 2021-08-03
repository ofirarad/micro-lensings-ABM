import numpy as np
import matplotlib.pyplot as plt


def gaussian_kernel(half_width):
    """

    :param half_width:
    :return:
    """
    if half_width == 0:
        # point source
        kernel = np.zeros((3, 3))
        kernel[1, 1] = 1
    else:
        # size larger than pixel
        temp_h, temp_v = np.meshgrid(np.arange(-3 * half_width, 3 * half_width + 1, 1),
                                     np.arange(-3 * half_width, 3 * half_width + 1, 1))
        kernel = np.exp(
            -(temp_h ** 2 + temp_v ** 2) / (2 * half_width ** 2))  # a Gaussian kernel for a finite point source
        kernel[kernel < np.exp(-3 ** 2 / 2)] = 0  # Kernel treats only cells within 3 sigma of centre
    return kernel


def path_finder(x0,y0,xf,yf):
    """
    This function returns a path based on initial and final positions in a 2D array
    :param x0: starting position
    :param y0:
    :param xf: final position
    :param yf:
    :return: numpy arrays x and y with values of the path
    """
    if x0==xf:
        y=np.arange(y0,yf+1,1,dtype=int)
        x=x0*np.ones(y.shape[0],dtype=int)
        return x,y
    if y0==yf:
        x = np.arange(x0, xf + 1, 1,dtype=int)
        y = y0 * np.ones(x.shape[0],dtype=int)
        return x,y
    x=np.array([x0])
    y=np.array([y0])
    slope=(yf-y0)/(xf-x0)
    intercept=(xf*y0-x0*yf)/(xf-x0)
    while x[-1]!=xf or y[-1]!=yf:
        xtemp,ytemp=np.meshgrid(np.array([x[-1]-1,x[-1],x[-1]+1]),np.array([y[-1]-1,y[-1],y[-1]+1]))
        dtemp=np.sqrt((xtemp*slope-ytemp+intercept)**2/(slope**2+1))+np.sqrt((xf-xtemp)**2+(yf-ytemp)**2)
        idx=np.unravel_index(np.argmin(dtemp, axis=None), dtemp.shape)
        x=np.append(x,xtemp[idx])
        y=np.append(y,ytemp[idx])
    return x,y


def path_conv(A,x0,y0,xf,yf,kernel,pixel2yr):
    """
    This function performs a convolution (actually cross-correlation) of the given kernel
    along path (x0,y0)->(xf,yf) in array A.
    :param A: a 2D numpy array
    :param kernel: a 2D kernel (numpy array), of which size is odd
    :pixel2yr: a coefficient to express the pixel size in years it takes to pass, assuming a typical transverse velocity and DL
    :return: convolved_path, a 1D array of the convolved values of kernel along the path
    """
    nx=A.shape[0] #number of rows in A
    ny=A.shape[1] #number of columns in A
    nkx=kernel.shape[0] # number of rows in kernel
    nky=kernel.shape[1] #number of columns in kernel
    dx=int(nkx // 2)
    dy=int(nky // 2)
    if dx+1+max(x0,xf)>nx or min(x0,xf)-dx<0 or dy+1+max(y0,yf)>ny or min(y0,yf)-dy<0:
        print('Kernel too large for chosen path')
        return 0
    norm=np.sum(kernel) # the normalization factor for each kernel
    xpath,ypath=path_finder(x0,y0,xf,yf)
    convolved_path=[]
    for t in range(xpath.shape[0]):
        temp_arr=A[xpath[t]-dx:xpath[t]+dx+1,ypath[t]-dy:ypath[t]+dy+1]
        temp_conv=np.sum(temp_arr*kernel)/norm
        convolved_path=np.append(convolved_path,temp_conv)
    time_path=pixel2yr*np.arange(0,convolved_path.shape[0])
    return time_path,convolved_path


def light_curve(half_width, resolution, FOV, mu_grid, vt_ein, shear_parallel=True,plot_flag=False):
    """
    This functions plots the light curve (either parallel or perpendicular motion) and returns the light curve array
    and the time-steps array in years.
    :param half_width: half-width of the point source, in pixels
    :param resolution: the 1-axis resolution of the FOV
    :param FOV: the Field of View in units of the typical Einstein radius
    :param mu_grid: the array of the source-plane magnification pattern
    :param vt_ein: the transverse velocity in units of Einstein radii per second
    :param shear_parallel: Choose between parallel (True) or perpendicular (False) path
    :return:
    """
    if shear_parallel:
        y0 = 3 * half_width + 1  # Initial and final position in pixels, x-rows, y-columns
        x0 = int(resolution / 2)
        yf = resolution - (3 * half_width + 2)
        xf = x0
    else:
        x0 = 3 * half_width + 1  # Initial and final position in pixels, x-rows, y-columns
        y0 = int(resolution / 2)
        xf = resolution - (3 * half_width + 2)
        yf = y0
    # vt_ein=the transverse velocity in Einstein radii per sec
    pixel2yr = ((FOV / resolution) / vt_ein) / (3600 * 24 * 365.25)

    WPK_kernel = gaussian_kernel(half_width)

    time_steps, light_curve_tmp = path_conv(mu_grid, x0, y0, xf, yf, WPK_kernel, pixel2yr)
    time_steps += pixel2yr * (
                3 * half_width + 1)  # to make sure light curves with different half-size match on the same axis
    if plot_flag:
        fig, ax = plt.subplots()
        plt.plot(time_steps, np.log10(np.abs(light_curve_tmp)))
        ax.set_xlabel('Time (yr)')
        ax.set_ylabel('Log $\mu$ (a.u.)')
    return time_steps, light_curve_tmp
