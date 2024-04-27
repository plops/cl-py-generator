#!/usr/bin/env python3
# python -m venv ~/pytorch_env
# . ~/pytorch_env/bin/activate
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu11
# pip install lmfit
import os
import time
import torch
import lmfit
class ImageModel(torch.nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        self.rgb_to_srgb_matrix=torch.nn.Parameter(torch.eye(3))
        self.brightness=torch.nn.Parameter(torch.tensor((1.0    )))
        self.offset=torch.nn.Parameter(torch.zeros(3))
        self.hue_angle=torch.nn.Parameter(torch.tensor((0.    )))
        self.gamma=torch.nn.Parameter(torch.tensor((2.20    )))
        self.rgb_to_yuv_matrix=torch.tensor([[(0.2990    ), (0.5870    ), (0.1140    )], [(-0.147130    ), (-0.288860    ), (0.4360    )], [(0.6150    ), (-0.514990    ), (-0.100010    )]])
    def forward(self, x):
        x=torch.matmul(x, self.rgb_to_srgb_matrix)
        x=torch.pow(x, (((1.0    ))/(self.gamma)))
        x=((x)*(self.brightness))
        x=((x)+(self.offset))
        x=torch.matmul(x, self.rgb_to_yuv_matrix)
        x=torch.matmul(x, torch.tensor([[1, 0, 0], [0, torch.cos(self.hue_angle), ((-1)*(torch.sin(self.hue_angle)))], [0, torch.sin(self.hue_angle), torch.cos(self.hue_angle)]]))
        return x
rgb_data=torch.rand(100, 3)
model=ImageModel()
initial_yuv=model(rgb_data)
target_yuv=((initial_yuv)+(((torch.randn_like(initial_yuv))*((0.10    )))))
def objective(params):
    # update model parameters from individual lmfit parameters
    model.rgb_to_srgb_matrix=torch.tensor([[params[f"rgb_to_srgb_matrix_{i}{j}"] for j in range(3)] for i in range(3)])
    model.brightness=params["brightness"]
    model.offset=torch.tensor([params[f"offset_{i}"] for i in range(3)])
    model.hue_angle=params["hue_angle"]
    model.gamma=params["gamma"]
    for param in model.parameters():
        param.requires_grad=True
    yuv_predicted=model(rgb_data)
    loss=torch.nn.functional_mse_loss(yuv_predicted, target_yuv)
    loss.backward()
    grads={name: param.grad.detach().numpy() for name, parm in model.named_parameters()}
    for param in model.parameters():
        param.requires_grad=False
    return loss
# define lmfit parameters
params=lmfit.Parameters()
for i in range(3):
    for j in range(3):
        params.add(f"rgb_to_srgb_matrix_{i}{j}", value=model.rgb_to_srgb_matrix[i,j].item())
params.add("brightness", value=model.brightness.item())
for i in range(3):
    params.add(f"offset", value=model.offset[i].item())
params.add("hue_angle", value=model.hue_angle.item())
params.add("gamma", value=model.gamma.item())
# run optimization with gradient information
result=lmfit.minimize(objective, params)