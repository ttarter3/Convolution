r"""
https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_network

https://visualstudio.microsoft.com/downloads/?cid=learn-navbar-download-cta

https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
conda install cuda -c nvidia

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import os
import numpy as np
import time


N = 6
data = [np.array((0))] * (2 * N)
if __name__ == "__main__":
    while (True):
        redraw = False
        
        for ii in range(N):
            data_file = r"../data/Filter.{}.bin".format(ii)
            if os.path.exists(data_file):
                data_new = np.fromfile(data_file, dtype=np.float32)
                data_new = data_new.reshape(int(np.sqrt(len(data_new))), int(np.sqrt(len(data_new))))
                if not np.array_equal(data_new, data[ii + N]):
                    redraw = True
                    data[ii + N] = data_new

            
            data_file = r"../data/Image.{}.bin".format(ii)
            if os.path.exists(data_file):
                data_new = np.fromfile(data_file, dtype=np.float32).reshape(1080, 1920)  
                if not np.array_equal(data_new, data[ii]):
                    redraw = True
                    data[ii] = data_new

        if redraw:
            fig = plt.figure()

            for ii in range(N):
                ax = fig.add_subplot(2, int(N / 2), ii + 1)
                ax.imshow(data[ii], cmap="gray")

                # ax = fig.add_subplot(2, N, ii + N + 1, projection='3d')
                # X, Y = np.meshgrid(range(data[ii + N].shape[0]), range(data[ii + N].shape[0]))
                # ax.plot_surface(X, Y, data[ii + N], rstride=1, cstride=1, linewidth=0, antialiased=False)                

            data_file = r"../data/Time.bin".format(ii)
            if os.path.exists(data_file):
                fig = plt.figure()
                plt.plot(np.fromfile(data_file, dtype=np.float32))

            plt.show()

        time.sleep(10)

    