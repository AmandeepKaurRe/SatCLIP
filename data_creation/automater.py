import os
import subprocess
import time


def complete_run(start, end):
    steps = list(range(start, end+1, 10000))
    ranges = [(steps[i], steps[i+1]) for i in range(len(steps)-1)]
    for s,e in ranges:
        run_submit_sbatch(s,e)
        time.sleep(3600)

def run_submit_sbatch(s,e):
        steps = list(range(s, e+1, 2000))
        ranges = [f'{steps[i]}-{steps[i+1]}' for i in range(len(steps)-1)]
        for r in ranges:
            submit_sbatch(r)

def submit_sbatch(args, script_path="/home/akaur64/GeoSSL/SATCLIP/data_creation/run_parallel.sh"):
    result = subprocess.run(
        ["sbatch", "--export", f"PYTHON_ARGS={args}", script_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Error submitting {script_path}: {result.stderr}")
    print(result.stdout.strip())
    return result.stdout.strip().split()[-1]  # Extract the job ID

start = 110000
end = 125000
# complete_run(start, end)
run_submit_sbatch(start, end)