import tensorflow
from tensorflow.python.client import device_lib
import sys, os, platform

if __name__ == '__main__':
    print(f"Tensorflow version: {tensorflow.__version__}")
    print(*device_lib.list_local_devices(), sep='\n')
    print(f"logical_devices (GPU): {tensorflow.config.list_logical_devices('GPU')}")
    print(f"physical_devices (GPU): {tensorflow.config.list_physical_devices('GPU')}")

    if platform.system() == "Linux":
        print(os.system("nvidia-smi"))
    elif platform.system() == "Windows":
        print(os.system(r"cd C:\Program Files\NVIDIA Corporation\NVSMI & nvidia-smi"))

    inp = tensorflow.keras.layers.Input(10)
    test_model = tensorflow.keras.models.Model(inp, tensorflow.keras.layers.Dense(10)(inp))
    test_model.compile()
    test_model.summary()
