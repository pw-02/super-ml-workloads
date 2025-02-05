import os
import subprocess
import sys
import time
from datetime import datetime

# Detect the correct Python version
def get_python_command():
    try:
        subprocess.run(["python", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return "python"
    except subprocess.CalledProcessError:
        try:
            subprocess.run(["python3", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return "python3"
        except subprocess.CalledProcessError:
            print("Error: Python is not installed.", file=sys.stderr)
            sys.exit(1)

# Define workload type and dataloader
workload_type = "image_classification"
dataloader = "coordl"

# Define workload configurations
workload_configs = ["cifar10_vit_b_16", "cifar10_vit_b_32", "cifar10_vit_l_16 ", "cifar10_vit_l_32 "]

# Define GPU indices and learning rates
job_ids = [0, 1, 2, 3]
learning_rates = [0.1, 0.01, 0.001, 0.0001]  # Add your learning rates here

# Generate experiment ID and log directory
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
expid = f"multi_job_{current_datetime}"
root_log_dir = "logs"
log_dir = os.path.join(root_log_dir, workload_type, dataloader, expid)
os.makedirs(log_dir, exist_ok=True)  # Ensure the log directory exists

# Start resource monitoring
print("Starting Resource Monitor...")
python_cmd = get_python_command()
monitor_cmd = f"{python_cmd} mlworkloads/resource_monitor.py start --interval 1 --flush_interval 10 --file_path {log_dir}/resource_usage_metrics.json"
with open(os.path.join(log_dir, "resource_monitor.log"), "w") as log_file:
    monitor_process = subprocess.Popen(monitor_cmd, shell=True, stdout=log_file, stderr=log_file)
monitor_pid = monitor_process.pid

# Track training start time
training_started_datetime = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
print(f"Training started UTC Time: {training_started_datetime}")

# Loop over jobs
job_pids = []
for i, workload in enumerate(workload_configs):
    workload = workload_configs[i]
    lr = learning_rates[i]
    print(f"Starting job on GPU {i} with workload {workload} and exp_id {expid}_{i}")
    run_cmd = f"CUDA_VISIBLE_DEVICES={i} {python_cmd} mlworkloads/run.py workload={workload} exp_id={expid} job_id={i} dataloader={dataloader} log_dir={log_dir}"
    #run_cmd = f"{python_cmd} mlworkloads/run.py workload={workload} exp_id={expid} job_id={jobid} dataloader={dataloader} log_dir={log_dir}"
    process = subprocess.Popen(run_cmd, shell=True)
    job_pids.append(process)
    time.sleep(2)  # Adjust as necessary

# Wait for all jobs to complete
for process in job_pids:
    process.wait()

# Track training end time
training_ended_datetime = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
print(f"Training started UTC Time: {training_started_datetime}")
print(f"Training ended UTC Time: {training_ended_datetime}")

# Stop resource monitor
print("Stopping Resource Monitor...")
monitor_process.terminate()

print("Experiment completed.")
