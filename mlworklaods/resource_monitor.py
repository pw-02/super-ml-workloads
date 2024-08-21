import os
import json
import time
import psutil
import subprocess
import threading

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
        self.thread = threading.Thread(target=self.collect_metrics, daemon=True)

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
            gpu_usage = self.get_gpu_usage()
            
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
        if self.metrics:
            # Append metrics to the file
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