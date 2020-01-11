import multiprocessing as mp
import os
import drprobe as drp
import sys
import numpy as np
import scipy.constants as constants
import tempfile
import shutil
from tqdm import tqdm_notebook as tqdm
from scipy.interpolate import interp1d


def create_empty_aber_dict():
    """
    Resets all aberrations to zero.
    """
    aberrations = {'A0': 0. + 1j * 0., 'A1': 0. + 1j * 0., 'C1': 0. + 1j * 0.,
                   'A2': 0. + 1j * 0., 'B2': 0. + 1j * 0., 'A3': 0. + 1j * 0.,
                   'C3': 0. + 1j * 0., 'S3': 0. + 1j * 0., 'A4': 0. + 1j * 0.,
                   'B4': 0. + 1j * 0., 'D4': 0. + 1j * 0., 'C5': 0. + 1j * 0.}

    return aberrations


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


class RonchiTrainer(object):
    """
    The number of simulated ronchigrams equals n_processes * n_batches * ly
    """

    def __init__(self, n_processes, n_batches, msa_obj, ht, nx, ny, ly, dd, mtf, thick_wobble=0):
        self.n_processes = n_processes
        self.n_batches = n_batches

        self._slc_dir, self._slc_file = msa_obj.slice_files.rsplit('/', 1)
        self._conv_semi_angle = msa_obj.conv_semi_angle
        self._a = msa_obj.h_scan_frame_size
        self._b = msa_obj.v_scan_frame_size
        self._nz = msa_obj.number_of_slices
        self._nt = msa_obj.tot_number_of_slices

        self._thick_wobble = thick_wobble

        self._ht = ht
        self._nx = nx
        self._ny = ny
        self._ly = ly
        self._dd = dd
        self._sinc = self._sinc_2d((self._ny, self._nx))
        self._mtf = self._mtf_2d((self._ny, self._nx), mtf)

        self._processes, self._pipes = [], []
        for proc_id in range(self.n_processes):
            master_conection, worker_connection = mp.Pipe(duplex=True)
            self._pipes.append(master_conection)

            p = mp.Process(name='worker_id{:02d}'.format(proc_id), target=self._worker,
                           args=(worker_connection,))
            self._processes.append(p)
            p.start()
            worker_connection.close()

    def finalize(self):
        for proc_id in range(self.n_processes):
            self._pipes[proc_id].send('STOP')
            self._pipes[proc_id].close()
            #print('worker_id{:02d} closed'.format(proc_id))
        for p in self._processes:
            p.join()

    def _worker(self, pipe):
        for arguments in iter(pipe.recv, 'STOP'):
            sys.stdout.flush()
            result = self._simulate_ronchi(*arguments)
            pipe.send(result)
        sys.stdout.flush()
        pipe.close()

    def _sinc_2d(self, shape):
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

    def _mtf_2d(self, shape, filename):
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

        mtf = np.loadtxt(filename, skiprows=1)
        func = interp1d(np.arange(0, mtf.shape[0], 1), mtf)

        x = np.abs(np.fft.fftfreq(int(shape[1])) * int(shape[1]))
        y = np.abs(np.fft.fftfreq(int(shape[0])) * int(shape[0]))

        xy = np.meshgrid(x, y)

        xxyy = np.hypot(xy[0], xy[1])
        xxyy = np.where(xxyy > mtf.shape[0] - 1, mtf.shape[0] - 1, xxyy)

        mtf_2d = func(xxyy.ravel()).reshape(shape)

        return mtf_2d

    def _shuffle_slices(self):
        headers, pots, file_names = {}, {}, {}
        for i, file in enumerate(os.listdir(self._slc_dir)):
            if file.endswith('.sli'):
                data = np.fromfile('{}/{}'.format(self._slc_dir, file), dtype='complex64')
                file_names[i] = file
                headers[i] = data[:1024]
                r_x = np.random.randint(0, self._nx)
                r_y = np.random.randint(0, self._ny)
                pots[i] = np.roll(data[1024:].reshape((self._nx, self._ny)), (r_x, r_y), (0, 1))

        keys = list(file_names.keys())
        np.random.shuffle(keys)
        for key in keys:
            data = np.concatenate((headers[key], pots[key].ravel()))
            data.astype('complex64').tofile('{}/{}'.format(self._slc_dir, file_names[key]))

    def _simulate_ronchi(self, aber_dict, px):
        #self.temp_path = tempfile.mkdtemp()
        with tempfile.TemporaryDirectory() as temp_path:
            msa_filename = os.path.join(temp_path, 'msa.prm')
            img_filename = os.path.join(temp_path, 'img.dat')

            # MSA
            msa = drp.MsaPrm()
            msa.conv_semi_angle = self._conv_semi_angle

            msa.focus_spread = 0
            msa.focus_spread_kernel_hw = 0
            msa.focus_spread_kernel_size = 0
            msa.spat_coherence_flag = 0
            msa.temp_coherence_flag = 0
            msa.source_radius = 0

            msa.wavelength = electron_wavelength(self._ht)
            msa.aberrations_dict = aber_dict
            msa.h_scan_frame_size = self._a
            msa.v_scan_frame_size = self._b
            msa.scan_columns = self._nx
            msa.scan_rows = self._ny
            msa.super_cell_x = 1
            msa.super_cell_y = 1
            msa.super_cell_z = 1
            msa.slice_files = f'{self._slc_dir}/{self._slc_file}'
            msa.number_of_slices = self._nz
            if self._thick_wobble != 0:
                nt = int(self._nt * ((np.random.random() - 0.5) * 2 * self._thick_wobble + 1))
            else:
                nt = self._nt
            msa.det_readout_period = nt
            msa.tot_number_of_slices = nt
            msa.save_msa_prm(msa_filename, random_slices=False)

            py = np.random.randint(0, self._ny - self._ly)
            drp.commands.msa(msa_filename, img_filename,
                             px=px,
                             py=py,
                             ly=py + self._ly - 1,
                             pdif=True)

            # Load data
            Y, pdifs = [], []
            for file in os.listdir(temp_path):
                if file.endswith('pdif_tot_sl{:03d}.dat'.format(nt)):
                    keys, idx = np.where(list(aber_dict.values()))
                    Y.append([aber_dict[key][i] for key, i in list(zip(keys, idx))])
                    #Y.append(f)
                    img = np.fromfile(os.path.join(temp_path, file), dtype='float32').reshape(
                        (self._nx, self._ny)).T
                    img_mtf_sinc = np.fft.fftshift(
                        np.real(np.fft.ifft2(np.fft.fft2(img) * self._sinc * self._mtf)))
                    pdifs.append(img_mtf_sinc[int((self._nx / 2) - self._dd):int(self._nx / 2 + self._dd),
                                 int((self._nx / 2) - self._dd):int(self._nx / 2 + self._dd)] / np.max(
                        img_mtf_sinc))

            X = np.array(pdifs)
            Y = np.array(Y)[:, np.newaxis]
            X = X[:, np.newaxis, :, :]

        #shutil.rmtree(self.temp_path)

        return X, Y

    def _convert_aber_dict(self, aber_dict):
        aberrations = {0: (np.real(aber_dict['A0']), np.imag(aber_dict['A0'])),
                       1: (np.real(aber_dict['C1']), 0),
                       2: (np.real(aber_dict['A1']), np.imag(aber_dict['A1'])),
                       3: (np.real(aber_dict['B2']), np.imag(aber_dict['B2'])),
                       4: (np.real(aber_dict['A2']), np.imag(aber_dict['A2'])),
                       5: (np.real(aber_dict['C3']), 0),
                       6: (np.real(aber_dict['S3']), np.imag(aber_dict['S3'])),
                       7: (np.real(aber_dict['A3']), np.imag(aber_dict['A3'])),
                       8: (np.real(aber_dict['B4']), np.imag(aber_dict['B4'])),
                       9: (np.real(aber_dict['D4']), np.imag(aber_dict['D4'])),
                       10: (np.real(aber_dict['A4']), np.imag(aber_dict['A4'])),
                       11: (np.real(aber_dict['C5']), 0)}
        return aberrations

    def generate_random_aberrations(self, aber_dict):
        """aber_dict contains the maximum aberration value"""
        dicts = []
        for i in range(self.n_processes):
            abd = create_empty_aber_dict()
            for key in aber_dict:
                if isinstance(aber_dict[key], (tuple, list, np.ndarray)):
                    ab_rnd = np.random.uniform(aber_dict[key][0], aber_dict[key][1], 2)
                elif isinstance(aber_dict[key], (float, int, complex)):
                    ab_rnd = (np.random.random(2) - np.array([0.5, 0.5])) * 2
                    ab_rnd *= np.array([np.abs(np.real(aber_dict[key])),
                                        np.abs(np.imag(aber_dict[key]))])
                abd[key] = ab_rnd[0] + 1j * ab_rnd[1]
            dicts.append(self._convert_aber_dict(abd))
        return dicts

    def __call__(self, aber_dict, verbose=False):
        xx, yy = [], []
        for i in tqdm(range(self.n_batches)):
            if verbose:
                print('Simulating batch #{} of Ronchigrams'.format(i))
            aber_dicts = self.generate_random_aberrations(aber_dict)
            self._shuffle_slices()
            pixel_x = np.arange(0, self._nx, 1)
            np.random.shuffle(pixel_x)
            chosen_px = pixel_x[:self.n_processes]

            for proc_id in range(self.n_processes):

                self._pipes[proc_id].send((aber_dicts[proc_id], chosen_px[proc_id]))

            X, Y = [], []
            for proc_id in range(self.n_processes):
                x, y = self._pipes[proc_id].recv()
                X.append(x)
                Y.append(y)
            X = np.vstack(X)
            Y = np.vstack(Y)

            xx.append(X)
            yy.append(Y)
        return np.vstack(xx), np.squeeze(np.vstack(yy))