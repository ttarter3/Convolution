r"""
https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_network

https://visualstudio.microsoft.com/downloads/?cid=learn-navbar-download-cta

https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
conda install cuda -c nvidia

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA
"""

import matplotlib.pyplot as plt
import os
import numpy as np
import time

if __name__ == "__main__":
    data_file = r"../data/output.img"

    data = np.array((0))

    while (True):
        if os.path.exists(data_file):
            data_new = np.fromfile(data_file, dtype=np.float32)
            if not np.array_equal(data_new, data):
                data = data_new

                working_data = data.reshape(1080, 1920)

                fig = plt.figure()
                plt.imshow(working_data, cmap="gray")

            plt.show()
        
        time.sleep(10)

    