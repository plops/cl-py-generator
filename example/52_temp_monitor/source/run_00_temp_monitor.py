import subprocess
import datetime
import time

_code_git_version = "0b601085e2cea058594b3bc9e7c0c04d314eb48f"
_code_repository = "https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py"
_code_generation_time = "07:59:45 of Monday, 2021-05-10 (GMT+1)"
count = 0
while (True):
    count = ((count) + (1))
    now = datetime.datetime.now()
    nowstr = now.strftime("%Y%m%d_%H%M_%S")
    with open("{}_sensors".format(nowstr), "w") as f:
        subprocess.call(["/usr/bin/sensors"], stdout=f)
    with open("{}_smart".format(nowstr), "w") as f:
        subprocess.call(["sudo", "/usr/sbin/smartctl", "-xa", "/dev/nvme0"],
                        stdout=f)
    with open("{}_nvme".format(nowstr), "w") as f:
        subprocess.call(["sudo", "/usr/sbin/nvme", "smart-log", "/dev/nvme0"],
                        stdout=f)
    with open("{}_nvda".format(nowstr), "w") as f:
        subprocess.call(["/opt/bin/nvidia-smi"], stdout=f)
    with open("{}_nvda2".format(nowstr), "w") as f:
        subprocess.call(["/opt/bin/nvidia-smi", "-q"], stdout=f)
    time.sleep(30)
    print(count)
