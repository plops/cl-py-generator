#|default_exp p00_torch_test
import os
import time
import datetime
import torch
start_time=time.time()
debug=True
_code_git_version="2e898672eb015020bf55c049a52fc80af10d72b3"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/104_rocm/source/"
_code_generation_time="14:37:38 of Saturday, 2023-03-11 (GMT+1)"
dev=torch.device("cuda")
x=torch.rand(1000, 1000, device=dev)
print("{} start ".format(((time.time())-(start_time))))
y=torch.mm(x, x)
print("{} end y={}".format(((time.time())-(start_time)), y))