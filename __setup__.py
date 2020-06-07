import tensorflow
from tensorflow.python.client import device_lib

if __name__ == '__main__':
    print(f"Tensorflow version: {tensorflow.__version__}")
    print(*device_lib.list_local_devices(), sep='\n')
    print(f"logical_devices (GPU): {tensorflow.config.list_logical_devices('GPU')}")
    print(f"physical_devices (GPU): {tensorflow.config.list_physical_devices('GPU')}")
