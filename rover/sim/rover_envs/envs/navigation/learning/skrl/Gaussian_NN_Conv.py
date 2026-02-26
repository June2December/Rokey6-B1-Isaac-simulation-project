"""Gaussian Neural Network Convolution encoder.

Provides GaussianNeuralNetworkConv and GaussianNNHeightmapEncoder
for use as heightmap encoders in skrl-compatible policy/value networks.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rover_envs.learning.models import get_activation


class GaussianNeuralNetworkConv(nn.Module):
    """
    Gaussian 커널을 활용한 신경망 컨볼루션 레이어.
    거리 기반 가중치 특징 추출에 최적화됨.
    """
    def __init__(self, in_channels, out_channels, sigma=1.0):
        super(GaussianNeuralNetworkConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 가우시안 대역폭(sigma)을 학습 가능한 파라미터로 설정 가능
        self.sigma = nn.Parameter(torch.tensor([sigma], dtype=torch.float32))

        # 특징 변환을 위한 선형 레이어
        self.projection = nn.Linear(in_channels, out_channels)

        # 초기화
        nn.init.xavier_uniform_(self.projection.weight)

    def forward(self, x, coords=None):
        """
        Args:
            x (torch.Tensor): 입력 특징 (batch, num_nodes, in_channels)
            coords (torch.Tensor): 각 특징의 좌표 정보 (batch, num_nodes, 1 or 2)
                                  없을 경우 인덱스 기반 거리를 사용합니다.
        """
        batch_size, num_nodes, _ = x.shape

        # 1. 특징 투영 (Linear Transformation)
        x_projected = self.projection(x)  # (batch, num_nodes, out_channels)

        # 2. 거리 행렬 계산 (좌표가 제공된 경우)
        if coords is not None:
            dist_sq = torch.cdist(coords, coords, p=2) ** 2  # (batch, num_nodes, num_nodes)
        else:
            indices = torch.arange(num_nodes, device=x.device).float()
            dist_sq = (indices.view(-1, 1) - indices.view(1, -1)) ** 2  # (num_nodes, num_nodes)

        # 3. Gaussian Kernel 적용: W = exp(-d^2 / (2 * sigma^2))
        kernel_matrix = torch.exp(-dist_sq / (2 * self.sigma ** 2))

        # 4. 가중치 정규화 (합이 1이 되도록)
        kernel_matrix = F.normalize(kernel_matrix, p=1, dim=-1)

        # 5. 가우시안 가중 합 (Convolution 연산)
        out = torch.matmul(kernel_matrix, x_projected)

        return F.leaky_relu(out)


class GaussianNNHeightmapEncoder(nn.Module):
    """
    Heightmap encoder using GaussianNeuralNetworkConv layers.

    ConvHeightmapEncoder(Conv2D) 대신 GaussianNeuralNetworkConv를 사용.
    Heightmap을 (H, H) 행렬로 reshape하여 각 행(row)을 하나의 노드로 처리.
    거리 정보를 Gaussian kernel로 인코딩하여 공간 관계를 학습.
    """
    def __init__(self, in_channels, encoder_features=[8, 16, 32, 64], encoder_activation="leaky_relu"):
        super().__init__()
        self.heightmap_size = int(math.sqrt(in_channels))

        self.gaussian_layers = nn.ModuleList()
        in_feat = self.heightmap_size
        for out_feat in encoder_features:
            self.gaussian_layers.append(GaussianNeuralNetworkConv(in_feat, out_feat))
            in_feat = out_feat

        compress_in = self.heightmap_size * encoder_features[-1]
        self.mlps = nn.ModuleList()
        self.mlps.append(nn.Linear(compress_in, 80))
        self.mlps.append(get_activation(encoder_activation))
        self.mlps.append(nn.Linear(80, 60))
        self.mlps.append(get_activation(encoder_activation))

        self.out_features = 60

    def forward(self, x):
        # x: (batch, H*H) flattened heightmap
        batch_size = x.size(0)
        H = self.heightmap_size

        x = x.view(batch_size, H, H)  # (batch, H, H)

        coords = torch.arange(H, device=x.device).float().unsqueeze(-1)   # (H, 1)
        coords = coords.unsqueeze(0).expand(batch_size, -1, -1)            # (batch, H, 1)

        for layer in self.gaussian_layers:
            x = layer(x, coords)  # (batch, H, out_feat)

        x = x.reshape(batch_size, -1)  # (batch, H * last_feat)

        for layer in self.mlps:
            x = layer(x)

        return x  # (batch, 60)
