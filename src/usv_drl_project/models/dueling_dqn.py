import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DuelingDQN(nn.Module):
    def __init__(self, input_shape, state_vec_dim, n_actions):
        super().__init__()
        c, h, w = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_output(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size + state_vec_dim, 256),
            nn.LayerNorm(256),  # 정규화 계층 추가
            nn.ReLU()
        )

        self.adv = nn.Linear(256, n_actions)
        self.val = nn.Linear(256, 1)

    def _get_conv_output(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, grid_map, state_vec):
        if grid_map.dtype != torch.float32:
            grid_map = grid_map.float() / 255.0  # 정규화 확실히 보장

        conv_out = self.conv(grid_map).view(grid_map.size(0), -1)
        x = torch.cat([conv_out, state_vec], dim=1)
        x = self.fc(x)
        adv = self.adv(x)
        val = self.val(x)
        return val + adv - adv.mean(dim=1, keepdim=True)

