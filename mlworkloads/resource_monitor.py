import os
import json
import time
import psutil
import subprocess
import threading
import numpy as np
class ResourceMonitor:
    def __init__(self, interval=1, flush_interval=10, file_path='resource_usage_metrics.json'):
        self.interval = interval
        self.flush_interval = flush_interval
        self.file_path = file_path
        self.metrics = []
        self.running = True
        self.start_time = time.time()
        self.total_cpu_usage = 0
        self.total_gpu_usage = 0
        self.count = 0
        self.last_flush_time = self.start_time
        self.gpu_available = self.check_gpu_availability()
        self.thread = threading.Thread(target=self.collect_metrics, daemon=True)
        
    def check_gpu_availability(self):
        try:
            result = subprocess.run(
                ['nvidia-smi', '--list-gpus'],
                capture_output=True, text=True
            )
            # If any GPUs are listed, we assume GPU is available
            return "GPU" in result.stdout
        except Exception as e:
            print(f"Error checking GPU availability: {e}")
            return False
        
    def start(self):
        self.thread.start()
    
    def stop(self):
        self.running = False
        self.thread.join()
        # Final flush to ensure all metrics are saved
        self.flush_metrics()
        # Optionally print or log final average metrics
        if self.count > 0:
            avg_cpu_usage = self.total_cpu_usage / self.count
            avg_gpu_usage = self.total_gpu_usage / self.count
            print(f"Final Average CPU Usage: {avg_cpu_usage:.2f}%")
            print(f"Final Average GPU Usage: {avg_gpu_usage:.2f}%")
        else:
            print("No metrics collected.")
    
    def collect_metrics(self):
        while self.running:
            cpu_usage = psutil.cpu_percent(interval=self.interval)
            gpu_usage = self.get_gpu_usage() if self.gpu_available else 0
            
            # Update totals and count
            self.total_cpu_usage += cpu_usage
            self.total_gpu_usage += gpu_usage
            self.count += 1
            
            # Calculate running averages
            avg_cpu_usage = self.total_cpu_usage / self.count
            avg_gpu_usage = self.total_gpu_usage / self.count
            
            elapsed_time = time.time() - self.start_time
            metrics = {
                "timestamp": time.time(),
                "elapsed_time": elapsed_time,  # Elapsed time since start
                "CPU Usage (%)": cpu_usage,
                "GPU Usage (%)": gpu_usage,
                "Average CPU Usage (%)": avg_cpu_usage,  # Running average
                "Average GPU Usage (%)": avg_gpu_usage   # Running average
            }
            self.metrics.append(metrics)
            
            # Flush metrics if the flush interval has passed
            current_time = time.time()
            if current_time - self.last_flush_time >= self.flush_interval:
                self.flush_metrics()
                self.last_flush_time = current_time
            
            time.sleep(self.interval)

    def get_gpu_usage(self):
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True
            )
            return int(result.stdout.strip())
        except Exception as e:
            print(f"Error retrieving GPU usage: {e}")
            return 0

    def flush_metrics(self):
        if os.path.exists(os.path.dirname(self.file_path)):
            if os.path.exists(self.file_path):
                with open(self.file_path, 'a') as f:
                    for metric in self.metrics:
                        f.write(json.dumps(metric) + "\n")
            else:
                with open(self.file_path, 'w') as f:
                    for metric in self.metrics:
                        f.write(json.dumps(metric) + "\n")
            # Clear the metrics list after flushing
            self.metrics.clear()

def cpu_stress_task(size):
       while True:
        # Perform matrix multiplication
        a = np.random.rand(size, size)
        b = np.random.rand(size, size)
        _ = np.dot(a, b)

def cpu_stress_test(thread_count, size, duration):
    threads = []
    end_time = time.time() + duration

    # Start multiple threads
    for _ in range(thread_count):
        thread = threading.Thread(target=cpu_stress_task, args=(size,))
        thread.start()
        threads.append(thread)

    # Run the stress test for the specified duration
    while time.time() < end_time:
        time.sleep(1)  # Sleep briefly to avoid high CPU usage by the main thread

    # Optionally, you can try to stop the threads gracefully (not always straightforward in Python)
    for thread in threads:
        thread.join(timeout=1)  # Wait for threads to complete

def test_cpu_load():
    
    thread_count = 1  # Number of threads to start
    matrix_size = 500  # Size of the matrix for matrix multiplication
    duration = 30  # Duration of the stress test in seconds

  # Start CPU stress test in a separate thread
    stress_thread = threading.Thread(target=cpu_stress_test, args=(thread_count, matrix_size, duration))
    stress_thread.start()

    # Run resource monitor alongside
    from resource_monitor import ResourceMonitor  # Adjust import based on file name
    monitor = ResourceMonitor(interval=1, flush_interval=5, file_path='resource_usage_metrics.json')

    print("Starting Resource Monitor and CPU Stress Test...")
    monitor.start()
    
    # Wait for the stress test to complete
    stress_thread.join()

    print("Stopping Resource Monitor...")
    monitor.stop()

    print("Test complete. Check 'resource_usage_metrics.json' for collected metrics.")

if __name__ == "__main__":
    test_cpu_load()
