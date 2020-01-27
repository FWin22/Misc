import numpy as np
import itertools
import math
import scalebars
import hyperspy.api as hs
from matplotlib.patches import Polygon
from scipy.spatial import Voronoi
import scipy.constants as constants
from scipy.ndimage.filters import minimum_filter, maximum_filter, median_filter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.interpolate import interp1d
from numpy import fft
from matplotlib.colors import LogNorm
from skimage import transform as tf
import scipy.optimize as opt
from Stuff import fitfunctions as ff
from scipy.ndimage.measurements import center_of_mass


def find_local_maxima(image, filter_order=25, border=1, map=False, mask=None):
    """
    Returns an array of all maxima coordinates.

    Parameters
    ----------
    image : ndarry
        The array on which the maxima-finder will operate
    filter_order : int, optional
        Defines the filter order for the maximum filter.
    border : int
        Defines the width of the border of the image, that will be neglected.
    map: bool, optional
        Returns an array of the image shape which is TRUE at positions of every local maximum.
    """

    # Indices of all pixels in the image
    yy, xx = np.indices(image.shape)
    local_max = maximum_filter(image, size=filter_order) == image

    # delete borders:
    borders = np.ones_like(image, dtype=np.bool)
    borders[border:-border, border:-border] = False

    maxima_array = np.logical_and(local_max, np.logical_not(borders))
    if mask is not None:
        maxima_array = np.where(mask, maxima_array, False)

    # Coordinates of all maxima
    maxima_x = xx[maxima_array]
    maxima_y = yy[maxima_array]
    max_coords = np.array(list(zip(maxima_y, maxima_x)))

    if map:
        return max_coords, maxima_array
    else:
        return max_coords


def find_local_minima(image, filter_order=25, border=1, map=False, mask=None):
    """
    Returns an array of all minimum coordinates.

    Parameters
    ----------
    image : ndarry
        The array on which the minimum-finder will operate
    filter_order : int, optional
        Defines the filter order for the minimum filter.
    border : int
        Defines the width of the border of the image, that will be neglected.
    map: bool, optional
        Returns an array of the image shape which is TRUE at positions of every local minimum.
    """

    # Indices of all pixels in the image
    yy, xx = np.indices(image.shape)
    local_min = minimum_filter(image, size=filter_order) == image

    # delete borders:
    borders = np.ones_like(image, dtype=np.bool)
    borders[border:-border, border:-border] = False

    minima_array = np.logical_and(local_min, np.logical_not(borders))

    if mask is not None:
        minima_array = np.where(mask, minima_array, False)

    # Coordinates of all maxima
    minima_x = xx[minima_array]
    minima_y = yy[minima_array]
    min_coords = np.array(list(zip(minima_y, minima_x)))

    if map:
        return min_coords, minima_array
    else:
        return min_coords


def rebin(a, shape):
    sh = (shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1])
    return a.reshape(sh).mean(-1).mean(1)


def normalize_image(image, roi=None):
    """
    Returns the image after normalization with respect to the reference given by roi.

    Parameters
    ----------
    image : ndarray
        The image which will be normalized. Can either be real or complex.
        If 'complex', the image will be divided by the mean of roi.
        If not 'complex', the mean of roi will be subtracted from the image.
    roi : ndarray or float or complex
        Is used to normalize the image by division for complex images and subtraction for real
        images. If roi is an array, the mean of this array will be subtracted from image. Hence
        it should be a masked array with np.nan outside the mask.
    """

    if np.issubdtype(image.dtype, np.complex):
        return image / np.nanmean(roi)
    else:
        return image - np.nanmean(roi)


def voronoi(coords):
    """
    Creates a Voronoi object.

    Parameters
    ----------
    coords : list of tuples
        A list containing all points (tuples) serving as sites for the Voronoi tesselation.
        """
    points = np.asarray(coords)
    vor = Voronoi(points)

    return vor


def voronoi_cell_average(vor, image, order):
    """
    Average the image within given Voronoi cells.
    Returns an array, with each pixel set to its corresponding mean value and a list containing
    the average value of each analyzed Voronoi cell

    Parameters
    ----------
    vor : scipy.spatial.Voronoi
        The voronoi object defining the cells in which the image will be averaged.
    image : ndarray
        The image that will be averaged within the Voronoi cells. Can be complex.
    order : int
        The number of expected vertices for regular patterns. Enter 0 for unregular patterns.
    """

    shape = image.shape
    n = len(vor.regions)

    if np.issubdtype(image.dtype, np.complex):
        image_voronoi = np.empty(shape, dtype='complex')
        nan = np.nan + 1j * np.nan
    else:
        image_voronoi = np.empty(shape)
        nan = np.nan

    image_voronoi[:] = nan
    pixelvalues_image, regionvalues_image, polys = [], [], []
    pixelregions = np.zeros(shape)

    for region in range(n):
        cell_sum, pixels_per_cell = 0, 0
        if validity(vor, vor.regions[region], shape, order):
            vertices = vor.vertices[vor.regions[region]]
            if len(vertices) > 2:
                x_max = int(np.ceil(vertices[:, 0].max()))
                x_min = int(np.floor(vertices[:, 0].min()))
                y_max = int(np.ceil(vertices[:, 1].max()))
                y_min = int(np.floor(vertices[:, 1].min()))

                poly = Polygon(np.fliplr(vor.vertices[vor.regions[region]]))

                polys.append(poly)
                for x, y in itertools.product(range(x_min, x_max), range(y_min, y_max)):

                    if poly.get_path().contains_point((y, x), radius=1) == 1:
                        pixels_per_cell += 1
                        cell_sum += image[x, y]
                        pixelregions[x, y] = region

        pixelvalues_image.append(cell_sum)
        if pixels_per_cell != 0:
            regionvalues_image.append(cell_sum / pixels_per_cell)
        else:
            regionvalues_image.append(nan)

    for y, x in itertools.product(range(0, shape[0]), range(0, shape[1])):
        h = int(pixelregions[y, x])
        if h != 0:
            image_voronoi[y, x] = regionvalues_image[h]
        else:
            image_voronoi[y, x] = nan

    return image_voronoi, regionvalues_image, polys


def validity(vor, vor_region, shape, n=0):
    """
    Checks if the voronoi region is as expected.

    Parameters
    ----------
    vor : scipy.spatial.Voronoi
        The voronoi object which contains the information.
    vor_region : scipy.spatial.Voronoi.region
        The special voronoi cell that is examined
    shape : (int, int)
        The dimensions of the image.
    n : int
        The number of expected vertices for regular patterns. Enter 0 for unregular patterns.
    """
    valid = True
    vertices = vor.vertices[vor_region]
    perimeter = 0

    # are all vertices in image?
    if valid:
        for vory, vorx in vertices:
            if vory < 0 or vory > shape[0] or vorx < 0 or vorx > shape[1]:
                valid = False

    if n == 0:
        return valid
    else:
        # check number of vertices
        if len(vor_region) != n:
            valid = False

        # calculate perimeter of hexagon
        for j in range(len(vertices)):
            u_x = vertices[j - 1, 0] - vertices[j, 0]
            u_y = vertices[j - 1, 1] - vertices[j, 1]
            u_len = math.sqrt(u_x * u_x + u_y * u_y)
            perimeter += u_len

        # check angles in hexagon.
        if valid:
            for k in range(len(vertices)):
                u_x = vertices[k - 1, 0] - vertices[k, 0]
                u_y = vertices[k - 1, 1] - vertices[k, 1]
                v_x = vertices[(k + 1) % n, 0] - vertices[k, 0]
                v_y = vertices[(k + 1) % n, 1] - vertices[k, 1]

                uv = np.abs(u_x * v_x + u_y * v_y)
                u_len = math.sqrt(u_x * u_x + u_y * u_y)
                v_len = math.sqrt(v_x * v_x + v_y * v_y)
                phi = 180 - math.degrees(math.acos(uv / (u_len * v_len)))
                alpha = (n - 2) / n * 180
                if phi < 0.8 * alpha or phi > 1.2 * alpha:
                    valid = False

        # check perimeter of hexagon
        if valid:
            for l in range(len(vertices)):
                u_x = vertices[l - 1, 0] - vertices[l, 0]
                u_y = vertices[l - 1, 1] - vertices[l, 1]
                u_len = math.sqrt(u_x * u_x + u_y * u_y)
                if u_len < 0.8 * perimeter / n or u_len > 1.2 * perimeter / n:
                    valid = False
    return valid


def freq_array(shape, sampling):
    """
    Returns an array with Fourier frequencies (in nm), corresponding to the given shape
    and sampling.

    Parameters
    ----------
    shape : tuple
        The shape of the array.
    sampling: tuple
        The sampling of the array given in nm (y, x).
    """

    f_freq_1d_y = np.fft.fftfreq(shape[0], sampling[0])
    f_freq_1d_x = np.fft.fftfreq(shape[1], sampling[1])
    f_freq_mesh = np.meshgrid(f_freq_1d_x, f_freq_1d_y)
    f_freq = np.hypot(f_freq_mesh[0], f_freq_mesh[1])

    return f_freq


def aperture_function(r, apradius, rsmooth=None):
    """
    A smooth aperture function that decays from apradius-rsmooth to apradius+rsmooth.

    Parameters
    ----------
        r : ndarray
            Array of input data (e.g. frequencies)
        apradius : float
            Radius (center) of the smooth aperture. Decay starts at apradius - rsmooth.
        rsmooth : float
            Smoothness in halfwidth. rsmooth = 1 will cause a decay from 1 to 0 over 2 pixel.
    """
    if rsmooth is None:
        rsmooth = 0.02 * apradius # This setting is identical to apertures in Dr. Probe
    return 0.5 * (1. - np.tanh((np.absolute(r) - apradius) / (0.5 * rsmooth)))


def rm_duds(img, sigma=8.0, median_k=5):
    """
    Removes dud pixels from images
    Parameters
    ----------
    img : ndarray
        The image
    sigma : float
    median_k : int
        Size of median kernel
    Returns
    -------
    img_nodud : ndarray
        Image with removed dud pixels (e.g. X-Rays spikes)
    Notes
    -----
    See Also
    --------
    """

    img_mf = median_filter(img, median_k)  # median filtered image
    diff_img = np.absolute(img - img_mf)
    mean_diff = sigma * np.std(diff_img)
    duds = diff_img > mean_diff
    img[duds] = img_mf[duds]

    # n_duds = np.sum(duds)  # dud pixels
    # print("The number of pixels changed = {}".format(n_duds))

    return img


def fold_array(arr, dim):
    """
    Folds an array by averaging.

    Parameter
    ---------

    arr : ndarray
        The array which is folded.
    dim : (int, int)
        Defines the number of multiple cells within the array. The output array has the
        following shape: arr.shape/dim.
    """
    folded_x = np.average(np.split(arr, dim[1], axis=0), axis=0) #/ dim[1]
    folded = np.average(np.split(folded_x, dim[0], axis=1), axis=0) #/ dim[0]

    return folded


def mean_min_distance(coords):
    """
    Returns the mean minimal distance in coordinate list.

    Parameters
    ----------
    coords : tuple list
        List of coordinates.
    """
    min_dist = []
    for i, ref in enumerate(coords):
        dist = []
        y0, x0 = ref
        for j, point in enumerate(coords):
            if i != j:
                y1, x1 = point
                d = np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
                dist.append(d)
        min_dist.append(np.min(dist))
    return np.min(min_dist)


def scalebar(ax, length, sampling, height=None, hidex=True, hidey=True, color='white',
             ecolor='black', loc=3, lw=0.5):
    """
    Provide height in pixels.
    """
    if height is None:
        height = length / (5 * sampling)

    scalebars.add_scalebar(ax, length/sampling, height, hidex=hidex, hidey=hidey,
                           color=color, ecolor=ecolor, loc=loc, lw=lw)


def sinc_2d(shape):
    """
    Returns a 2D sinc function for the given shape.

    Parameters
    ----------
    shape : tuple
        The shape for the sinc function.
    """
    # Sinc(f_Ny) = 0.64, Sinc(2*f_Ny) = 0
    y = np.fft.fftfreq(int(shape[0]), 0.5)
    x = np.fft.fftfreq(int(shape[1]), 0.5)
    xy = np.meshgrid(x, y)
    sinc = (np.sinc(xy[0] / 2) * np.sinc(xy[1] / 2))

    return sinc


def mtf_2d(shape, filename=None):
    """
    Returns the 2D MTF for the PICO microscope for a given camera binning.

    Parameters
    ----------
    shape : int, list, tuple, ndarray
        Provide the image size.
    """
    if not isinstance(shape, (np.ndarray, list, tuple)):
        shape = (shape, shape)
    nyq = int(shape[0] / 2)
    binning = int(4096 / (2 * nyq))

    if filename is not None:
        mtf = np.loadtxt(filename, skiprows=1)
    else:
        mtf = np.loadtxt('MTF/PICO-US4k-080_mtf_bin{}_{}.mtf'.format(binning, 2 * nyq), skiprows=1)
    func = interp1d(np.arange(0, mtf.shape[0], 1), mtf)

    x = np.abs(np.fft.fftfreq(int(shape[1])) * int(shape[1]))
    y = np.abs(np.fft.fftfreq(int(shape[0])) * int(shape[0]))

    xy = np.meshgrid(x, y)

    xxyy = np.hypot(xy[0], xy[1])
    xxyy = np.where(xxyy > mtf.shape[0] - 1, mtf.shape[0] - 1, xxyy)

    mtf_2d = func(xxyy.ravel()).reshape(shape)

    return mtf_2d


def convolve_2d(arr1, arr2):
    """
    Returns the convolution between arr1 and arr2. The convolution is performed via multiplication in Fourier space.
    """
    fft_arr1 = np.fft.fft2(arr1)
    fft_arr2 = np.fft.fft2(arr2)

    return np.real(np.fft.ifft2(fft_arr1 * fft_arr2))


def radial_profile(data, center):
    """
    Returns radial profie of the data.

    Parameters
    ----------
    data : ndarray
        Input data
    center: tuple, list
        Center for radial profile. Provide as (y, x).
    """
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr

    return radialprofile


def transform_array(signal, min_coords, dim, x_new, y_new, sampling,
                    left_to_right=True, verbose=False):
    """
    Transforms subregions of the given data to new frame. The lattice coordinates from
    min_coords are mapped onto equivalent points of a regular lattice with constants x_new and
    y_new.

    Parameters
    ----------
    signal : hs.signal.ComplexSignal2D
        Hyperspy signal of the full wave function
    min_coords : list, np.array
        Contains the measured lattice points that.
    dim : tuple
        The dimensionality of the coordinates, e.g how many unit cells (x, y)
    x_new : int
        Length of the new unit cell in x-direction
    y_new : int
        Length of the new unit cell in y-direction
    sampling : tuple
        The sampling rates of the original signal (y, x)
    left_to_right : bool, optional
        Are the coordinates probe the image from left to rigth, or vice versa.
    verbose : bool, optional
        Additional output of the rotation angle.
    """
    waves_ud = []
    for coords in min_coords:

        dst = np.fliplr(coords)
        src = []

        if left_to_right:
            for i in range(len(dst)):
                u = (i % (dim[0] + 1)) * x_new
                v = (i // (dim[0] + 1)) * y_new
                src.append([u, v])
            src = np.asarray(src)
        else:
            for i in range(len(dst)):
                u = ((dim[0] - i) % (dim[0] + 1)) * x_new
                v = (i // (dim[0] + 1)) * y_new
                src.append([u, v])
            src = np.asarray(src)

        # tform = tf.ProjectiveTransform()
        tform = tf.AffineTransform()
        tform.estimate(src, dst)

        if verbose:
            print('Rotation angle: {:.2f}°'.format(np.degrees(tform.rotation)))

        shape = np.multiply((y_new, x_new), (dim[1], dim[0]))

        if np.issubdtype(signal.data.dtype, np.dtype(complex)):
            warped_real = tf.warp(signal.data.real, tform, output_shape=shape, preserve_range=True)
            warped_imag = tf.warp(signal.data.imag, tform, output_shape=shape, preserve_range=True)
            warped_com = warped_real + 1j * warped_imag
            wave_ud = hs.signals.ComplexSignal2D(warped_com)
        elif np.issubdtype(signal.data.dtype, np.dtype(float)):
            warped = tf.warp(signal.data, tform, output_shape=shape, preserve_range=True)
            wave_ud = hs.signals.Signal2D(warped)
        elif np.issubdtype(signal.data.dtype, np.dtype(int)):
            warped = tf.warp(signal.data, tform, output_shape=shape, preserve_range=True)
            wave_ud = hs.signals.Signal2D(warped)

        wave_ud.metadata = signal.metadata.copy()
        wave_ud.axes_manager[0].scale = sampling[1]
        wave_ud.axes_manager[0].size = shape[1]
        wave_ud.axes_manager[0].units = 'nm'
        wave_ud.axes_manager[0].name = 'x'
        wave_ud.axes_manager[1].scale = sampling[0]
        wave_ud.axes_manager[1].size = shape[0]
        wave_ud.axes_manager[1].units = 'nm'
        wave_ud.axes_manager[1].name = 'y'
        waves_ud.append(wave_ud)

    new_waves = hs.stack(waves_ud)

    return new_waves


def plot_subregions_ud(signal, sampling, labels, nrows, n_cols, mode='phase', clim=(0, 1),
                       figsize=(18, 10), cmap='gray',
                       axes_pad=0.35, cbar_mode='single', cbar_pad=0.25, x_min=None, x_max=None,
                       y_min=None, y_max=None, lognorm=False, cb_label=None):
    fig = plt.figure(figsize=figsize)

    grid = ImageGrid(fig, 111, nrows_ncols=(nrows, n_cols), axes_pad=axes_pad, share_all=True,
                     cbar_location="right", cbar_mode=cbar_mode, cbar_size="5%", cbar_pad=cbar_pad)

    ims = []

    for i, ax in enumerate(grid):
        if mode == 'phase':
            ims.append(ax.imshow(np.angle(signal.inav[i].data), clim=clim, cmap=cmap))
        elif mode == 'amplitude':
            ims.append(ax.imshow(np.abs(signal.inav[i].data), clim=clim, cmap=cmap))
        elif mode == 'fftabs':
            if lognorm:
                ims.append(ax.imshow(
                    np.abs(np.fft.fftshift(signal.inav[i].data))[y_min:y_max, x_min:x_max],
                    norm=LogNorm(vmin=clim[0], vmax=clim[1]),
                    # clim=(0, 0.2),
                    cmap=cmap, interpolation='none'))
            else:
                ims.append(ax.imshow(
                    np.abs(np.fft.fftshift(signal.inav[i].data))[y_min:y_max, x_min:x_max],
                    clim=clim, cmap=cmap, interpolation='none'))
        elif mode == 'fft':
            if lognorm:
                ims.append(ax.imshow(np.fft.fftshift(signal.inav[i].data)[y_min:y_max, x_min:x_max],
                    norm=LogNorm(vmin=clim[0], vmax=clim[1]),
                    # clim=(0, 0.2),
                    cmap=cmap, interpolation='none'))
            else:
                ims.append(ax.imshow(np.fft.fftshift(signal.inav[i].data)[y_min:y_max, x_min:x_max],
                    clim=clim, cmap=cmap, interpolation='none'))

        # scalebars 0.2 nm
        if mode == 'phase' or mode == 'amplitude':
            scalebar(ax, 0.2, np.mean(sampling))
        elif mode == 'fft' or mode == 'fftabs':
            scalebar(ax, 2, np.mean(sampling))
        ax.set_title(labels[i])

        # Colorbar
        if mode == 'phase':
            cb = ax.cax.colorbar(ims[i])
        elif mode == 'amplitude':
            cb = ax.cax.colorbar(ims[i])
        elif mode == 'fft' or mode == 'fftabs':
            if lognorm:
                cb = ax.cax.colorbar(ims[i], ticks=[0.01, 0.1, 1])
                cb.ax.set_yticklabels([0.01, 0.1, 1])
            else:
                cb = ax.cax.colorbar(ims[i])
        ax.cax.toggle_label(True)

        if cb_label is None:
            if mode == 'phase':
                cb_label = 'phase shift [rad]'
            elif mode == 'amplitude':
                cb_label = 'amplitude'
            elif mode == 'fft' or mode == 'fftabs':
                cb_label = 'amplitude'

        cb.set_label_text(cb_label)

    return fig


def create_fft_signal(signal, shape, f_sampling):
    ffts = []
    for i, s in enumerate(signal):
        fft_c = hs.signals.ComplexSignal2D(fft.fftn(s.data) / np.prod(shape))
        fft_c.metadata = signal.metadata.copy()
        fft_c.axes_manager[0].scale = f_sampling[0]
        fft_c.axes_manager[0].size = shape[0]
        fft_c.axes_manager[0].units = '1/nm'
        fft_c.axes_manager[0].name = 'y'
        fft_c.axes_manager[1].scale = f_sampling[1]
        fft_c.axes_manager[1].size = shape[1]
        fft_c.axes_manager[1].units = '1/nm'
        fft_c.axes_manager[1].name = 'x'
        ffts.append(fft_c)

    fft_signal = hs.stack(ffts)
    return fft_signal


def beam_amplitude_thresholds(fft_data, min_dist=2, maxiter=25):
    def cost(t):
        pos_x, pos_y = np.where(np.absolute(np.fft.fftshift(fft_data)) > t)
        pos = list(zip(pos_x, pos_y))

        if len(pos) > 1:
            mins = []
            for i, p in enumerate(pos):
                dists = [np.sqrt((p[0] - pos[j][0]) ** 2 + (p[1] - pos[j][1]) ** 2) for j in
                         range(0, len(pos))]
                dists.pop(i)
                mins.append(np.min(dists))

            if np.min(mins) < min_dist:
                return 0
            else:
                return 1

        else:
            return 1

    bounds = [np.min(np.abs(fft_data)), np.max(np.abs(fft_data))]

    for i in range(maxiter):
        if bounds[1] - bounds[0] > 1e-5:
            pass
        x = np.mean(bounds)
        s = cost(x)
        if s == 0:
            bounds[0] = np.floor(x * 1e6) / 1e6
        elif s == 1:
            bounds[1] = np.ceil(x * 1e6) / 1e6

    return np.ceil(x * 1e4) / 1e4


def nmoment(img, n, center_origin=False, sampling=None, ref=None):
    """
    Calculate n-th moment of input data.
    Parameters
    ----------
    img : array
        input data
    n : int
        Order of moment
    center_origin : bool, optional
        Shift the origin of the coordinate system to the center of the image.
    sampling : tuple, optional
        Return output in terms of image coordinates, rather than pixels. sampling rate (y, x)
    """
    y, x = np.indices(img.shape, dtype='float')
    if center_origin:
        y -= img.shape[0] / 2 #- 0.5
        x -= img.shape[1] / 2 #- 0.5
    if sampling is not None:
        y *= sampling[0]
        x *= sampling[1]
    if np.sum(img) == 0:
        return [np.nan, np.nan]
    else:
        if ref is None:
            return [np.sum((y)**n * img) / np.sum(img), np.sum((x)**n * img) / np.sum(img)]
        else:
            return [np.sum((y)**n * img) / np.sum(ref), np.sum((x)**n * img) / np.sum(ref)]


def mrad_to_reciprocal_nm(alpha, ht=80):
    """
    Convert scattering angles from mrad to reciprocal nm.
    Parameters
    ----------
    alpha : float
        Angle in mrad.
    ht : float, optional
        High tension in kV.
    """
    return alpha / (1000 * electron_wavelength(ht))


def reciprocal_nm_to_mrad(f, ht=80):
    """
    Convert frequency from reciprocal nm to mrad.
    Parameters
    ----------
    f : float
        Frequency in reciprocal nm.
    ht : float, optional
        High tension in kV.
    """
    return f * 1000 * electron_wavelength(ht)


def normalize(signal, mask, poly_order_y, poly_order_x, mode='phase'):
    nn = '{}x{}y'.format(poly_order_x, poly_order_y)
    if mode == 'complex':
        nn += 'c'

    shape = signal.data.shape
    x = np.linspace(0, 1, shape[1])
    y = np.linspace(0, 1, shape[0])

    x, y = np.meshgrid(x, y)
    xm = x[mask]
    ym = y[mask]

    com_vac = np.array(center_of_mass(mask))
    xv = np.linspace(0, 1, shape[1]) - (com_vac[1] / shape[1])
    yv = np.linspace(0, 1, shape[0]) - (com_vac[0] / shape[0])

    xv, yv = np.meshgrid(xv, yv)

    if mode == 'phase':
        data = np.angle(signal.data)
        initial_guess = [0] * len(ff._polynomial_params(poly_order_x, poly_order_y))
        popt, pcov = opt.curve_fit(ff.polynomial(poly_order_x, poly_order_y), (ym, xm), data[mask],
                                   p0=initial_guess)
        ramp = np.abs(np.mean(signal.data[mask])) * np.exp(1j *
                                                           ff.polynomial(poly_order_x,
                                                                         poly_order_y)((y, x),
                                                                                       *popt))

        sigma_ramp = 1 * np.exp(1j * ff.polynomial_sigma(poly_order_x, poly_order_y)((yv, xv),
                                                                                     *np.sqrt(
                                                                                         np.diag(
                                                                                             pcov))))

    elif mode == 'complex':
        data_real = np.real(signal.data)
        initial_guess = [0] * len(ff._polynomial_params(poly_order_x, poly_order_y))
        popt_real, pcov_real = opt.curve_fit(ff.polynomial(poly_order_x, poly_order_y), (ym, xm),
                                             data_real[mask],
                                             p0=initial_guess)
        data_imag = np.imag(signal.data)
        initial_guess = [0] * len(ff._polynomial_params(poly_order_x, poly_order_y))
        popt_imag, pcov_imag = opt.curve_fit(ff.polynomial(poly_order_x, poly_order_y), (ym, xm),
                                             data_imag[mask],
                                             p0=initial_guess)

        ramp = (ff.polynomial(poly_order_x, poly_order_y)((y, x), *popt_real) + 1j *
                ff.polynomial(poly_order_x, poly_order_y)((y, x), *popt_imag))

        sigma_ramp = (ff.polynomial_sigma(poly_order_x, poly_order_y)((yv, xv), *np.sqrt(
            np.diag(pcov_real))) + 1j *
                      ff.polynomial_sigma(poly_order_x, poly_order_y)((yv, xv), *np.sqrt(
                          np.diag(pcov_imag))))

        popt = popt_real + 1j * popt_imag
        pcov = pcov_real + 1j * pcov_imag

    elif mode == 'complex2':
        data_abs = np.abs(signal.data)
        initial_guess = [0] * len(ff._polynomial_params(poly_order_x, poly_order_y))
        popt_abs, pcov_abs = opt.curve_fit(ff.polynomial(poly_order_x, poly_order_y), (ym, xm),
                                             data_abs[mask],
                                             p0=initial_guess)
        data_angle = np.angle(signal.data)
        initial_guess = [0] * len(ff._polynomial_params(poly_order_x, poly_order_y))
        popt_angle, pcov_angle = opt.curve_fit(ff.polynomial(poly_order_x, poly_order_y), (ym, xm),
                                             data_angle[mask],
                                             p0=initial_guess)

        ramp = (ff.polynomial(poly_order_x, poly_order_y)((y, x), *popt_abs) * np.exp(1j *
                ff.polynomial(poly_order_x, poly_order_y)((y, x), *popt_angle)))

        sigma_ramp = (ff.polynomial_sigma(poly_order_x, poly_order_y)((yv, xv), *np.sqrt(
            np.diag(pcov_abs))) * np.exp(1j *
                      ff.polynomial_sigma(poly_order_x, poly_order_y)((yv, xv), *np.sqrt(
                          np.diag(pcov_angle)))))

        popt = popt_abs * np.exp(1j * popt_angle)
        pcov = pcov_abs * np.exp(1j * pcov_angle)

    return popt, pcov, ramp, sigma_ramp


def figsize(scale, height=None, textwidth=448.1309):
    """
    Calculates ideal matplotlib Figure size, according to the desirde scale.
    :param scale: Fraction of Latex graphic input (scale*\textwidth)
    :param height: figure height = figure width * height
    :param textwidth:
    :return:
    """
    fig_width_pt = textwidth                         # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27                       # Convert pt to inch
    golden_mean = constants.golden_ratio - 1            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale    # width in inches
    if height is None:
        fig_height = fig_width * golden_mean              # height in inches
    else:
        fig_height = fig_width * height
    fig_size = [fig_width, fig_height]

    return fig_size


def savefig(filename, formats=None, pgf=True, bbox_inches=None, pad_inches=None, dpi=300,
            pdf=True):
    if formats is None:
        if pgf:
            plt.savefig('{}.pgf'.format(filename), bbox_inches=bbox_inches)
        if pdf:
            plt.savefig('{}.pdf'.format(filename), bbox_inches=bbox_inches, dpi=dpi, pad_inches=pad_inches)
    else:
        if type(formats) is list or type(formats) is tuple:
            for fmt in formats:
                plt.savefig('{}.{}'.format(filename, fmt),
                            bbox_inches=bbox_inches, dpi=dpi, pad_inches=pad_inches)
    #plt.savefig('{}.png'.format(filename), bbox_inches=bbox_inches, dpi=dpi,
                # pad_inches=pad_inches)


def electron_wavelength(ht):
    """
    Returns electron wavelenght in nm.
    Parameters
    ----------
    ht : float
        High tension in kV.
    """
    ec = constants.elementary_charge
    me = constants.m_e
    momentum = 2 * me * ec * ht * 1000 * (1 + ec * ht * 1000 / (2 * me * constants.c ** 2))
    wavelength = constants.h / np.sqrt(momentum) * 1e9  # in nm

    return wavelength


def interaction_constant(ht):
    """
    Returns the interaction constant C_E.
    Parameters
    ----------
    ht : float
        High tension in kV.
    """
    ec = constants.elementary_charge
    me = constants.m_e

    E0 = me * constants.speed_of_light**2
    E = ec * ht * 1e3 * (2 * E0 + ec * ht * 1e3) / (2 * (E0 + ec * ht * 1e3))
    CE = np.pi * ec / (electron_wavelength(ht) * 1e-9 * E)

    return CE


def electron_velocity(ht):
    """
    Returns the relativistic electron velocity in m/s.
    Parameters
    ----------
    ht : float
        High tension in kV.
    """
    return constants.elementary_charge / (constants.hbar * interaction_constant(ht))


def significant_round(x):
    if x != 0:
        sign_exp = np.int(np.ceil(-np.log10(np.abs(x))))
        last_sign_digit = np.int(np.round(x * 10**sign_exp))
        return last_sign_digit / 10**sign_exp
    else:
        return 0


def histogram(x, bins, density=True):
    hist, bin_edges = np.histogram(x, bins=bins, density=density)
    step = bin_edges[1] - bin_edges[0]
    bin_center = bin_edges[:-1] + 0.5 * step
    return hist, bin_center, step


def gaussian_noise(amplitude, aperture=40, shape=(100, 100), sampling=(1, 1),
                   wavelength=0.004, ntype='normal'):
    if ntype == 'uniform':
        noise_r = np.random.uniform(-amplitude, amplitude, shape)
        noise_i = np.random.uniform(-amplitude, amplitude, shape)
    elif ntype == 'normal':
        noise_r = np.random.normal(0, amplitude, shape)
        noise_i = np.random.normal(0, amplitude, shape)
    else:
        noise_r = np.zeros(shape)
        noise_i = np.zeros(shape)

    if aperture != 0:
        f_freq = freq_array(shape, sampling)
        ap = aperture_function(f_freq, aperture / (wavelength * 1000), 1 / (wavelength * 1000))

        noise_r_fft = np.fft.fftn(noise_r) / np.prod(shape) * ap
        noise_r = np.real(np.fft.ifftn(noise_r_fft) * np.prod(shape))

        noise_i_fft = np.fft.fftn(noise_i) / np.prod(shape) * ap
        noise_i = np.real(np.fft.ifftn(noise_i_fft) * np.prod(shape))

        noise = noise_r + 1j * noise_i
    else:
        noise = noise_r + 1j * noise_i
    return noise


def find_closest_prime(x, nmax=5):
    numbers = {}
    for i in range(nmax):
        for j in range(nmax):
            for k in range(nmax):
                index = (i, j, k)
                numbers[index] = 2 ** i * 3 ** j * 5 ** k

    k = list(numbers.keys())
    v = np.array(list(numbers.values()))
    dist = abs(v - x)
    arg = np.argmin(dist)
    return v[arg]


def find_closest_int_with_low_prime_factors(x):

    numbers = []
    for i in range(10):
        for j in range(10):
            for k in range(10):
                numbers.append(2**i * 3**j * 5**k)

    numbers = np.sort(numbers)
    numbers = numbers[:len(np.where(numbers < 100000)[0])]

    dist = np.abs(numbers - x)
    return numbers[np.argmin(dist)]


def pretty_colorbar(fig, axs, im, position='top', pad=0.1, width=0.03, cbar_label=None,
                    constrain_ticklabels=False, ticks=None, ticklabels=None):
    """
    Creates a pretty colorbar, aligned with figure axes.

    Parameters
    ----------
    fig : matplotlib.figure object
        The figure object that contains the matplotlib axes and artists.
    axs : matplotlib.axes or list of matplotlib.axes
        The axes object(s), where the colorbar is drawn.
        Only provide those axes, which the colorbar will span.
    im : matplotlib object, mappable
        Mappable matplotlib object.
    position : str, optional
        The position defines the location of the colorbar. One of 'top', 'bottom', 'left' or
        'right'.
    pad : float, optional
        Defines the spacing between the axes and colorbar axis. Is given in figure fraction.
    width : float, optional
        Width of the colorbar given in figure fraction.
    cbar_label : string, optional
        Colorbar label
    constrain_ticklabels : bool, optional
        Allows to slightly shift the outermost ticklabels, such that they do not exceed the cbar
        axis.
    ticks : list, np.ndarray, optional
        List of cbar ticks
    ticklabels : list, np.ndarray, optional
        List of cbar ticklabels
    -----------
    Returns the colorbar object
    """
    if isinstance(axs, (tuple, list, np.ndarray)):
        p = []
        for ax in axs:
            p.append(ax.get_position().get_points().flatten())
    else:
        p = [axs.get_position().get_points().flatten()]

    if position == 'top':
        ax_cbar = fig.add_axes([p[0][0], 1-pad, p[-1][2]-p[0][0], width])
        cb = plt.colorbar(im, cax=ax_cbar, orientation='horizontal')
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.xaxis.set_label_position('top')
    elif position == 'bottom':
        ax_cbar = fig.add_axes([p[0][0], pad, p[-1][2] - p[0][0], width])
        cb = plt.colorbar(im, cax=ax_cbar, orientation='horizontal')
        cb.ax.xaxis.set_ticks_position('bottom')
        cb.ax.xaxis.set_label_position('bottom')
    elif position == 'right':
        ax_cbar = fig.add_axes([1-pad, p[-1][1], width, p[0][3] - p[-1][1]])
        cb = plt.colorbar(im, cax=ax_cbar, orientation='vertical')
        cb.ax.yaxis.set_ticks_position('right')
        cb.ax.yaxis.set_label_position('right')
    elif position == 'left':
        ax_cbar = fig.add_axes([pad, p[-1][1], width, p[0][3] - p[-1][1]])
        cb = plt.colorbar(im, cax=ax_cbar, orientation='vertical')
        cb.ax.yaxis.set_ticks_position('left')
        cb.ax.yaxis.set_label_position('left')

    # Colorbar label
    if cbar_label is not None:
        cb.set_label(r'{}'.format(cbar_label))

    # Ticks and ticklabels
    if ticks:
        cb.set_ticks(ticks)
    if ticklabels:
        cb.set_ticklabels(ticklabels)

    # Constrain tick labels
    if constrain_ticklabels:
        if position == 'top' or position == 'bottom':
            t = cb.ax.get_xticklabels()
            t[0].set_horizontalalignment('left')
            t[-1].set_horizontalalignment('right')
        elif position == 'left' or position == 'right':
            t = cb.ax.get_yticklabels()
            t[0].set_verticalalignment('top')
            t[-1].set_verticalalignment('bottom')

    return cb
