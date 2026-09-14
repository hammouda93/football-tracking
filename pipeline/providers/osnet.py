# OSNet portions: Copyright (c) Kaiyang Zhou, licensed under the MIT License.
# See THIRD_PARTY_NOTICES.md.
"""Small OSNet x0.25 inference adapter for native Windows Re-ID.

The network definition is adapted from deep-person-reid's MIT-licensed OSNet
implementation (Kaiyang Zhou et al.).  Training and Cython ranking utilities are
intentionally not included: the football pipeline only needs checkpoint loading
and batched feature extraction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        return self.relu(self.bn(self.conv(inputs)))


class Conv1x1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        return self.relu(self.bn(self.conv(inputs)))


class Conv1x1Linear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, inputs):
        return self.bn(self.conv(inputs))


class LightConv3x3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            padding=1,
            bias=False,
            groups=out_channels,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        return self.relu(self.bn(self.conv2(self.conv1(inputs))))


class ChannelGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = channels // reduction
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=True)
        self.gate_activation = nn.Sigmoid()

    def forward(self, inputs):
        gates = self.global_avgpool(inputs)
        gates = self.fc2(self.relu(self.fc1(gates)))
        return inputs * self.gate_activation(gates)


class OSBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        middle = out_channels // 4
        self.conv1 = Conv1x1(in_channels, middle)
        self.conv2a = LightConv3x3(middle, middle)
        self.conv2b = nn.Sequential(
            LightConv3x3(middle, middle),
            LightConv3x3(middle, middle),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(middle, middle),
            LightConv3x3(middle, middle),
            LightConv3x3(middle, middle),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(middle, middle),
            LightConv3x3(middle, middle),
            LightConv3x3(middle, middle),
            LightConv3x3(middle, middle),
        )
        self.gate = ChannelGate(middle)
        self.conv3 = Conv1x1Linear(middle, out_channels)
        self.downsample = (
            Conv1x1Linear(in_channels, out_channels)
            if in_channels != out_channels
            else None
        )

    def forward(self, inputs):
        identity = inputs if self.downsample is None else self.downsample(inputs)
        scale_input = self.conv1(inputs)
        combined = (
            self.gate(self.conv2a(scale_input))
            + self.gate(self.conv2b(scale_input))
            + self.gate(self.conv2c(scale_input))
            + self.gate(self.conv2d(scale_input))
        )
        return F.relu(self.conv3(combined) + identity)


class OSNet(nn.Module):
    def __init__(self, *, num_classes: int = 1, feature_dim: int = 512):
        super().__init__()
        channels = [16, 64, 96, 128]
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(channels[0], channels[1], reduce=True)
        self.conv3 = self._make_layer(channels[1], channels[2], reduce=True)
        self.conv4 = self._make_layer(channels[2], channels[3], reduce=False)
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels[3], feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)
        self._init_params()

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int, *, reduce: bool):
        layers: list[nn.Module] = [
            OSBlock(in_channels, out_channels),
            OSBlock(out_channels, out_channels),
        ]
        if reduce:
            layers.append(
                nn.Sequential(
                    Conv1x1(out_channels, out_channels),
                    nn.AvgPool2d(2, stride=2),
                )
            )
        return nn.Sequential(*layers)

    def _init_params(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, inputs):
        outputs = self.maxpool(self.conv1(inputs))
        outputs = self.conv5(self.conv4(self.conv3(self.conv2(outputs))))
        outputs = self.global_avgpool(outputs).flatten(1)
        return self.fc(outputs)


def _torch_device(value: str):
    normalized = str(value or "cpu").strip().lower()
    if normalized in {"", "cpu", "none"}:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA demandee pour OSNet mais indisponible")
    if normalized.isdigit():
        normalized = f"cuda:{normalized}"
    elif normalized == "cuda":
        normalized = "cuda:0"
    return torch.device(normalized)


class OSNetFeatureExtractor:
    """Load an official OSNet x0.25 checkpoint and embed OpenCV BGR crops."""

    def __init__(self, model_path: str, device: str = "cpu"):
        checkpoint_path = Path(model_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        self.device = _torch_device(device)
        self.model = OSNet(num_classes=1)
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:  # PyTorch before the weights_only argument
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        else:
            raise ValueError("checkpoint OSNet non reconnu")
        model_state = self.model.state_dict()
        compatible = {}
        for key, value in state_dict.items():
            key = str(key)
            if key.startswith("module."):
                key = key[7:]
            if (
                key in model_state
                and hasattr(value, "shape")
                and model_state[key].shape == value.shape
            ):
                compatible[key] = value
        if len(compatible) < 100:
            raise ValueError(
                f"checkpoint OSNet incompatible: {len(compatible)} couches reconnues"
            )
        model_state.update(compatible)
        self.model.load_state_dict(model_state)
        self.model.eval().to(self.device)
        self.matched_layers = len(compatible)

    def __call__(self, crops: list[np.ndarray]):
        import cv2

        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        tensors = []
        for crop in crops:
            if crop.size == 0:
                crop = np.zeros((256, 128, 3), dtype=np.uint8)
            resized = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensors.append(((rgb - mean) / std).transpose(2, 0, 1))
        batch = torch.from_numpy(np.asarray(tensors, dtype=np.float32)).to(self.device)
        with torch.inference_mode():
            return self.model(batch)
