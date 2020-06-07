import tensorflow
from tensorflow.python.client import device_lib

if __name__ == '__main__':
    print(f"Tensorflow version: {tensorflow.__version__}")
    device_lib.list_local_devices()
