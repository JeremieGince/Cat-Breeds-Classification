import tensorflow
from tensorflow.python.client import device_lib
import sys, os, platform

if __name__ == '__main__':
    print(f"Tensorflow version: {tensorflow.__version__} \n")
    print(*device_lib.list_local_devices(), '\n', sep='\n')

    if tensorflow.__version__ == "2.2.0":
        print(f"logical_devices (GPU): {tensorflow.config.list_logical_devices('GPU')} \n")
        print(f"physical_devices (GPU): {tensorflow.config.list_physical_devices('GPU')} \n")

    if platform.system() == "Linux":
        print(os.system("nvidia-smi"), '\n')
    elif platform.system() == "Windows":
        print(os.system(r"cd C:\Program Files\NVIDIA Corporation\NVSMI & nvidia-smi"), '\n')
