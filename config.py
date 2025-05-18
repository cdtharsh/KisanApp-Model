import os
import tensorflow as tf

# Set environment variables
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# Set GPU memory limit if GPU is available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
        )
        print("Memory limit set to 4GB")
    except RuntimeError as e:
        print("GPU memory config error:", e)

CONFIDENCE_THRESHOLD = 99.0
