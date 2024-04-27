#!/usr/bin/env python3
# python -m venv ~/pytorch_env
# . ~/pytorch_env/bin/activate
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu11
import os
import time
import torch
import pandas as pd
import lmfit
class ImageModel(torch.nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        self.rgb_to_srgb_matrix=torch.nn.Parameter(torch.eye(3))
        self.brightness=torch.nn.Parameter(torch.tensor((1.0    )))
        self.offset=torch.nn.Parameter(torch.zeros(3))
        self.hue_angle=torch.nn.Parameter(torch.tensor((0.    )))
        self.gamma=torch.nn.Parameter(torch.(2.20    ))
        self.rgb_to_yuv_matrix=torch.tensor([[(0.2990    ), (0.5870    ), (0.1140    )], [(-0.147130    ), (-0.288860    ), (0.4360    )], [(0.6150    ), (-0.514990    ), (-0.100010    )]])
    def forward(self, x):
        x=torch.matmul(x, self.rgb_to_srgb_matrix)
        x=torch.pow(x, (((1.0    ))/(self.gamma)))
        x=((x)*(self.brightness))
        x=((x)+(self.offset))
        x=torch.matmul(x, rgb_to_yuv_matrix)
        x=torch.matmul(x, torch.tensor([[1, 0, 0], [0, torch.cos(self.hue_angle), ((-1)*(torch.sin(self.hue_angle)))], [0, torch.sin(self.hue_angle), torch.cos(self.hue_angle)]]))
        return x