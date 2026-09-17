"""Self-contained architecture builders adapted for RIPPLe.

The executable family definitions are adapted from DeepLense AI Scientist
revision ``dbea8485f8bf64250e2aec82dd8999d354085996``.  RIPPLe does not import
or require that repository at runtime. Source files reviewed for this port:

* ``dlens/tools/_torch_backends.py``
  SHA-256 ``78e1916d24f7d6c979a60262efb88227be53af30fb651c829d0b4e4dfeae7be0``
* ``dlens/tools/_arch_families.py``
  SHA-256 ``8563bb5a0808870bc9c7de3f241c75fa22db5a9e9f0ab6daca164ead41da6b3b``

The port changes the schema boundary to RIPPLe's strict ``BoundArchitecture``
and keeps only model construction. Dataset loading, CUDA selection, training,
metrics, and checkpoint handling live in the RIPPLe backend.
"""

from __future__ import annotations

import math
from typing import Any

from ..schemas.architecture import ArchitectureFamily, BoundArchitecture


AI_SCIENTIST_SOURCE_REVISION = "dbea8485f8bf64250e2aec82dd8999d354085996"
RIPPLE_MODEL_BUILDERS_REVISION = "ripple-model-builders-v1"


def _torch_modules() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch import nn
        import torch.nn.functional as functional
    except ImportError:
        raise RuntimeError("RIPPLe model construction requires PyTorch") from None
    return torch, nn, functional


def _heads_for(dimension: int) -> int:
    for heads in (8, 4, 2):
        if dimension % heads == 0:
            return heads
    return 1


def _patch_for(size: int) -> int:
    for patch in (16, 15, 14, 12, 10, 8, 6, 5, 4):
        if size % patch == 0:
            return patch
    return 15


def build_model(architecture: BoundArchitecture, *, dropout: float = 0.0) -> Any:
    """Build one of the six agent-visible families from a code-bound contract."""

    torch, nn, functional = _torch_modules()
    candidate = architecture.candidate
    family = candidate.family
    depths = tuple(candidate.depths)
    widths = tuple(candidate.widths)
    input_channels = architecture.channels
    class_count = architecture.num_classes
    height, width = architecture.input_shape

    if family == ArchitectureFamily.CNN:
        layers: list[Any] = []
        channels = input_channels
        for stage_depth, stage_width in zip(depths, widths, strict=True):
            for _ in range(stage_depth):
                layers.extend(
                    (
                        nn.Conv2d(channels, stage_width, 3, 1, 1, bias=False),
                        nn.BatchNorm2d(stage_width),
                        nn.ReLU(inplace=True),
                    )
                )
                channels = stage_width
            layers.append(nn.MaxPool2d(2))
        layers.extend(
            (
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(widths[-1], class_count),
            )
        )
        return nn.Sequential(*layers)

    if family == ArchitectureFamily.RESNET:

        class BasicBlock(nn.Module):
            def __init__(self, incoming: int, outgoing: int, stride: int = 1) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(incoming, outgoing, 3, stride, 1, bias=False)
                self.bn1 = nn.BatchNorm2d(outgoing)
                self.conv2 = nn.Conv2d(outgoing, outgoing, 3, 1, 1, bias=False)
                self.bn2 = nn.BatchNorm2d(outgoing)
                self.activation = nn.ReLU(inplace=True)
                self.downsample = None
                if stride != 1 or incoming != outgoing:
                    self.downsample = nn.Sequential(
                        nn.Conv2d(incoming, outgoing, 1, stride, bias=False),
                        nn.BatchNorm2d(outgoing),
                    )

            def forward(self, inputs: Any) -> Any:
                identity = (
                    inputs if self.downsample is None else self.downsample(inputs)
                )
                outputs = self.activation(self.bn1(self.conv1(inputs)))
                outputs = self.bn2(self.conv2(outputs))
                return self.activation(outputs + identity)

        stem: list[Any] = [
            nn.Conv2d(input_channels, widths[0], 7, 2, 3, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        ]
        stages: list[Any] = []
        incoming = widths[0]
        for index, (stage_depth, stage_width) in enumerate(
            zip(depths, widths, strict=True)
        ):
            blocks = [BasicBlock(incoming, stage_width, 1 if index == 0 else 2)]
            blocks.extend(
                BasicBlock(stage_width, stage_width) for _ in range(stage_depth - 1)
            )
            stages.append(nn.Sequential(*blocks))
            incoming = stage_width
        head = [
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(widths[-1], class_count),
        ]
        return nn.Sequential(*stem, *stages, *head)

    class PatchEmbed(nn.Module):
        def __init__(self, incoming: int, dimension: int, patch: int) -> None:
            super().__init__()
            self.patch = patch
            self.projection = nn.Conv2d(
                incoming, dimension, kernel_size=patch, stride=patch
            )

        def forward(self, inputs: Any) -> Any:
            pad_height = (-inputs.shape[-2]) % self.patch
            pad_width = (-inputs.shape[-1]) % self.patch
            if pad_height or pad_width:
                inputs = functional.pad(inputs, (0, pad_width, 0, pad_height))
            return self.projection(inputs).flatten(2).transpose(1, 2)

    class TransformerBlock(nn.Module):
        def __init__(self, dimension: int, heads: int) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(dimension)
            self.attention = nn.MultiheadAttention(
                dimension, heads, dropout=dropout, batch_first=True
            )
            self.norm2 = nn.LayerNorm(dimension)
            hidden = max(dimension, dimension * 4)
            self.mlp = nn.Sequential(
                nn.Linear(dimension, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, dimension),
            )

        def forward(self, inputs: Any) -> Any:
            normalized = self.norm1(inputs)
            outputs = (
                inputs
                + self.attention(
                    normalized, normalized, normalized, need_weights=False
                )[0]
            )
            return outputs + self.mlp(self.norm2(outputs))

    if family == ArchitectureFamily.VIT:
        dimension = widths[-1]
        block_count = max(1, sum(depths))
        patch = _patch_for(height)
        token_count = math.ceil(height / patch) * math.ceil(width / patch)

        class VisionTransformer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = PatchEmbed(input_channels, dimension, patch)
                self.class_token = nn.Parameter(torch.zeros(1, 1, dimension))
                self.position = nn.Parameter(torch.zeros(1, token_count + 1, dimension))
                nn.init.trunc_normal_(self.class_token, std=0.02)
                nn.init.trunc_normal_(self.position, std=0.02)
                self.blocks = nn.ModuleList(
                    TransformerBlock(dimension, _heads_for(dimension))
                    for _ in range(block_count)
                )
                self.norm = nn.LayerNorm(dimension)
                self.dropout = nn.Dropout(dropout)
                self.head = nn.Linear(dimension, class_count)

            def forward(self, inputs: Any) -> Any:
                outputs = self.embedding(inputs)
                class_tokens = self.class_token.expand(outputs.shape[0], -1, -1)
                outputs = torch.cat((class_tokens, outputs), dim=1)
                outputs = outputs + self.position[:, : outputs.shape[1]]
                for block in self.blocks:
                    outputs = block(outputs)
                return self.head(self.dropout(self.norm(outputs)[:, 0]))

        return VisionTransformer()

    if family == ArchitectureFamily.MLPMIXER:
        dimension = widths[-1]
        block_count = max(1, sum(depths))
        patch = _patch_for(height)
        token_count = math.ceil(height / patch) * math.ceil(width / patch)

        class MixerBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm1 = nn.LayerNorm(dimension)
                self.token_mixer = nn.Sequential(
                    nn.Linear(token_count, max(8, token_count // 2)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(max(8, token_count // 2), token_count),
                )
                self.norm2 = nn.LayerNorm(dimension)
                self.channel_mixer = nn.Sequential(
                    nn.Linear(dimension, dimension * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dimension * 2, dimension),
                )

            def forward(self, inputs: Any) -> Any:
                mixed = self.norm1(inputs).transpose(1, 2)
                outputs = inputs + self.token_mixer(mixed).transpose(1, 2)
                return outputs + self.channel_mixer(self.norm2(outputs))

        class Mixer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = PatchEmbed(input_channels, dimension, patch)
                self.blocks = nn.ModuleList(MixerBlock() for _ in range(block_count))
                self.norm = nn.LayerNorm(dimension)
                self.dropout = nn.Dropout(dropout)
                self.head = nn.Linear(dimension, class_count)

            def forward(self, inputs: Any) -> Any:
                outputs = self.embedding(inputs)
                for block in self.blocks:
                    outputs = block(outputs)
                return self.head(self.dropout(self.norm(outputs).mean(dim=1)))

        return Mixer()

    if family == ArchitectureFamily.HYBRID:
        convolution_depths = depths[:-1]
        convolution_widths = widths[:-1]
        dimension = widths[-1]
        attention_blocks = max(1, depths[-1])

        class Hybrid(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                layers: list[Any] = []
                incoming = input_channels
                for stage_depth, stage_width in zip(
                    convolution_depths, convolution_widths, strict=True
                ):
                    for _ in range(stage_depth):
                        layers.extend(
                            (
                                nn.Conv2d(incoming, stage_width, 3, 1, 1, bias=False),
                                nn.BatchNorm2d(stage_width),
                                nn.ReLU(inplace=True),
                            )
                        )
                        incoming = stage_width
                    layers.append(nn.MaxPool2d(2))
                self.stem = nn.Sequential(*layers)
                self.projection = nn.Conv2d(incoming, dimension, 1)
                self.position = nn.Parameter(torch.zeros(1, 64, dimension))
                nn.init.trunc_normal_(self.position, std=0.02)
                self.blocks = nn.ModuleList(
                    TransformerBlock(dimension, _heads_for(dimension))
                    for _ in range(attention_blocks)
                )
                self.norm = nn.LayerNorm(dimension)
                self.dropout = nn.Dropout(dropout)
                self.head = nn.Linear(dimension, class_count)

            @staticmethod
            def _fixed_grid(inputs: Any) -> Any:
                target_height = math.ceil(inputs.shape[-2] / 8) * 8
                target_width = math.ceil(inputs.shape[-1] / 8) * 8
                if (target_height, target_width) != inputs.shape[-2:]:
                    inputs = functional.pad(
                        inputs,
                        (
                            0,
                            target_width - inputs.shape[-1],
                            0,
                            target_height - inputs.shape[-2],
                        ),
                    )
                return functional.avg_pool2d(
                    inputs,
                    kernel_size=(target_height // 8, target_width // 8),
                )

            def forward(self, inputs: Any) -> Any:
                outputs = self._fixed_grid(self.projection(self.stem(inputs)))
                outputs = outputs.flatten(2).transpose(1, 2) + self.position
                for block in self.blocks:
                    outputs = block(outputs)
                return self.head(self.dropout(self.norm(outputs).mean(dim=1)))

        return Hybrid()

    if family == ArchitectureFamily.EQUIVARIANT:

        class C4Conv(nn.Module):
            def __init__(self, incoming: int, outgoing: int, *, lifting: bool) -> None:
                super().__init__()
                self.incoming = incoming
                self.outgoing = outgoing
                self.lifting = lifting
                shape = (
                    (outgoing, incoming, 3, 3)
                    if lifting
                    else (outgoing, incoming, 4, 3, 3)
                )
                self.weight = nn.Parameter(torch.empty(*shape))
                nn.init.kaiming_normal_(
                    self.weight.view(outgoing, -1, 3, 3),
                    mode="fan_out",
                    nonlinearity="relu",
                )
                self.bias = nn.Parameter(torch.zeros(outgoing))

            def forward(self, inputs: Any) -> Any:
                if self.lifting:
                    rotations = [
                        torch.rot90(self.weight, rotation, dims=(2, 3))
                        for rotation in range(4)
                    ]
                    weights = torch.stack(rotations, dim=1).reshape(
                        self.outgoing * 4, self.incoming, 3, 3
                    )
                else:
                    rotations = []
                    for rotation in range(4):
                        rotated = torch.rot90(self.weight, rotation, dims=(3, 4))
                        rotations.append(torch.roll(rotated, shifts=rotation, dims=2))
                    weights = torch.stack(rotations, dim=1).reshape(
                        self.outgoing * 4, self.incoming * 4, 3, 3
                    )
                return functional.conv2d(
                    inputs,
                    weights,
                    self.bias.repeat_interleave(4),
                    padding=1,
                )

        class OrientationBatchNorm(nn.Module):
            def __init__(self, channels: int) -> None:
                super().__init__()
                self.channels = channels
                self.normalization = nn.BatchNorm3d(channels)

            def forward(self, inputs: Any) -> Any:
                batch, _, rows, columns = inputs.shape
                outputs = inputs.view(batch, self.channels, 4, rows, columns)
                return self.normalization(outputs).view(
                    batch, self.channels * 4, rows, columns
                )

        class Equivariant(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                layers: list[Any] = []
                incoming = input_channels
                lifting = True
                for stage_depth, stage_width in zip(depths, widths, strict=True):
                    for _ in range(stage_depth):
                        layers.extend(
                            (
                                C4Conv(incoming, stage_width, lifting=lifting),
                                OrientationBatchNorm(stage_width),
                                nn.ReLU(inplace=True),
                            )
                        )
                        incoming = stage_width
                        lifting = False
                    layers.append(nn.MaxPool2d(2))
                self.body = nn.Sequential(*layers)
                self.channels = incoming
                self.multiple = 2 ** len(depths)
                self.dropout = nn.Dropout(dropout)
                self.head = nn.Linear(incoming, class_count)

            def _symmetric_pad(self, inputs: Any) -> Any:
                vertical = (-inputs.shape[-2]) % self.multiple
                horizontal = (-inputs.shape[-1]) % self.multiple
                top, bottom = vertical // 2, vertical - vertical // 2
                left, right = horizontal // 2, horizontal - horizontal // 2
                return (
                    functional.pad(inputs, (left, right, top, bottom))
                    if any((left, right, top, bottom))
                    else inputs
                )

            def forward(self, inputs: Any) -> Any:
                outputs = self.body(self._symmetric_pad(inputs))
                batch, _, rows, columns = outputs.shape
                outputs = outputs.view(batch, self.channels, 4, rows, columns).amax(
                    dim=2
                )
                outputs = functional.adaptive_avg_pool2d(outputs, 1).flatten(1)
                return self.head(self.dropout(outputs))

        return Equivariant()

    raise ValueError(f"unsupported architecture family: {family.value}")


__all__ = [
    "AI_SCIENTIST_SOURCE_REVISION",
    "RIPPLE_MODEL_BUILDERS_REVISION",
    "build_model",
]
