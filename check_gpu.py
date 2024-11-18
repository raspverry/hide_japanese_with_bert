# check_gpu.py
import os

import spacy


print("Environment:")
print(f"CUDA_HOME: {os.getenv('CUDA_HOME')}")
print(f"PATH: {os.getenv('PATH')}")
print(f"LD_LIBRARY_PATH: {os.getenv('LD_LIBRARY_PATH')}")
print("\nspaCy info:")
print(f"spaCy version: {spacy.__version__}")

# GPU backend check
try:
	from thinc.api import prefer_gpu, require_gpu
	from thinc.util import gpu_is_available

	print(f"GPU available: {gpu_is_available()}")
	print(f"Prefer GPU: {prefer_gpu()}")
	require_gpu()
	print("\nGPU requirement enforced")
except Exception as e:
	print(f"\nGPU check failed: {e}")

# CuPy check
try:
	import cupy

	print("\nCuPy info:")
	print(f"CuPy CUDA version: {cupy.cuda.runtime.runtimeGetVersion()}")
	print(f"Available GPU devices: {cupy.cuda.runtime.getDeviceCount()}")
except Exception as e:
	print(f"\nCuPy check failed: {e}")
