import os
import time
import pynvml
import matplotlib.pyplot as plt

# Initialize NVML (NVIDIA Management Library)
pynvml.nvmlInit()

def get_gpu_info():
    # Get GPU handle
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    
    # GPU name
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    
    # Memory info
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total_memory = mem_info.total / (1024 ** 2)  # Convert bytes to MB
    used_memory = mem_info.used / (1024 ** 2)    # Convert bytes to MB
    free_memory = mem_info.free / (1024 ** 2)    # Convert bytes to MB

    # Utilization rates
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    gpu_util = utilization.gpu  # GPU utilization percentage

    return gpu_name, total_memory, used_memory, free_memory, gpu_util

def monitor_gpu(interval=1, duration=60):
    gpu_utilizations = []
    
    for _ in range(duration // interval):
        gpu_name, total_memory, used_memory, free_memory, gpu_util = get_gpu_info()
        gpu_utilizations.append(gpu_util)
        print(f"GPU: {gpu_name}")
        print(f"Memory: {used_memory:.2f} MB / {total_memory:.2f} MB")
        print(f"GPU Utilization: {gpu_util}%")
        time.sleep(interval)
    
    return gpu_utilizations

def plot_gpu_utilization(utilizations):
    plt.plot(utilizations)
    plt.title("GPU Utilization Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Utilization (%)")
    plt.show()

if __name__ == "__main__":
    duration = 60  # Monitor for 60 seconds
    interval = 1   # Check every 1 second
    
    print("Starting GPU monitoring...")
    gpu_utilizations = monitor_gpu(interval=interval, duration=duration)
    
    print("Plotting GPU utilization data...")
    plot_gpu_utilization(gpu_utilizations)
    
    # Shutdown NVML
    pynvml.nvmlShutdown()
