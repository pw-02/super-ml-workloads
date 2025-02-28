import os
import subprocess
import sys
import time
from datetime import datetime, timezone

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


#job speeds to tested
job_speeds_list = [
    [1.01, 0.99, 1.00, 1.02],       # Very Low Variability (Range ≈ 0.03)
    [1.05, 0.97, 1.02, 0.98],       # Low Variability (Range ≈ 0.08)
    [0.90, 1.00, 1.10, 1.00],       # Mild Variability (Range ≈ 0.20)
    [0.80, 0.90, 1.20, 1.10],       # Moderate Variability (Range ≈ 0.40)
    [0.70, 0.85, 1.30, 1.15],       # Medium-High Variability (Range ≈ 0.60)
    [0.60, 0.80, 1.40, 1.20],       # High Variability (Range ≈ 0.80)
    [0.50, 0.70, 1.60, 1.30],       # Very High Variability (Range ≈ 1.10)
    [0.40, 0.65, 1.70, 1.45],       # Extreme Variability (Range ≈ 1.30)
    [0.35, 0.60, 1.80, 1.50],       # Ultra-Extreme Variability (Range ≈ 1.45)
    [0.30, 0.55, 2.00, 1.60]        # Maximum Variability (Range ≈ 1.70)
]

run_id = 0

job_speeds = job_speeds_list[run_id]

# Define workload type and dataloader
workload_type = "scalability_varying_speeds"
dataset = "imagenet"
dataloader = "super" #super, coordl #baseline

# Define workload configurations
workload_configs = ["imagenet_resnet18", "imagenet_resnet50", "imagenet_resnet18", "imagenet_resnet18"]

# Define GPU indices and learning rates
job_ids = [0, 1, 2, 3]
learning_rates = [0.1, 0.01, 0.001, 0.0001]  # Add your learning rates here

# Generate experiment ID and log directory
current_datetime = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
expid = f"multi_job_{current_datetime}"
root_log_dir = "logs"
log_dir = os.path.join(root_log_dir, workload_type, dataset, dataloader, expid)
os.makedirs(log_dir, exist_ok=True)  # Ensure the log directory exists

# Start resource monitoring
print("Starting Resource Monitor...")
python_cmd = get_python_command()
monitor_cmd = f"{python_cmd} mlworkloads/resource_monitor.py start --interval 1 --flush_interval 10 --file_path {log_dir}/resource_usage_metrics.json"
with open(os.path.join(log_dir, "resource_monitor.log"), "w") as log_file:
    monitor_process = subprocess.Popen(monitor_cmd, shell=True, stdout=log_file, stderr=log_file)
monitor_pid = monitor_process.pid

# Track training start time
training_started_datetime =  datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
print(f"Training started UTC Time: {training_started_datetime}")

# Loop over jobs
job_pids = []
for i, workload in enumerate(workload_configs):
    workload = workload_configs[i]
    lr = learning_rates[i]
    job_speed = job_speeds[i]
    print(f"Starting job on GPU {i} with job speed {job_speed} and exp_id {expid}_{i}")
    run_cmd = f"CUDA_VISIBLE_DEVICES={i} {python_cmd} mlworkloads/run.py workload={workload} exp_id={expid} job_id={i} dataloader={dataloader} log_dir={log_dir} workload.num_pytorch_workers=2 workload.gpu_time={job_speed} simulation_mode=True"
    #run_cmd = f"{python_cmd} mlworkloads/run.py workload={workload} exp_id={expid} job_id={jobid} dataloader={dataloader} log_dir={log_dir}"
    process = subprocess.Popen(run_cmd, shell=True)
    job_pids.append(process)
    time.sleep(2)  # Adjust as necessary

# Wait for all jobs to complete
for process in job_pids:
    process.wait()

# Track training end time
training_ended_datetime =  datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
print(f"Training started UTC Time: {training_started_datetime}")
print(f"Training ended UTC Time: {training_ended_datetime}")

# Stop resource monitor
print("Stopping Resource Monitor...")
monitor_process.kill()

print("Experiment completed.")
