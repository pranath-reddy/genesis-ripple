"""Pure-PyTorch inference runtime for the recovered Mriganka ENN checkpoints.

The training implementation used equivariant convolutions.  An evaluation-mode
checkpoint also stores each equivariant convolution's expanded, ordinary
``Conv2d`` filter.  This module consumes those stored filters directly and
reproduces the inference graph with PyTorch tensor operations.  It therefore
has no runtime dependency on ``e2cnn`` or on the researcher repository.

This module deliberately does not read checkpoint files.  The caller is
responsible for safe deserialization (for example, ``torch.load`` with
``weights_only=True``), integrity verification, and device selection before
constructing these modules.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, TypeAlias

import torch
from torch import Tensor, nn
from torch.nn import functional as F

INPUT_CHANNELS: Final = 3
INPUT_HEIGHT: Final = 64
INPUT_WIDTH: Final = 64
REGULAR_REPRESENTATION_SIZE: Final = 8
LATENT_SIZE: Final = 256
LOGIT_COUNT: Final = 2

_BATCH_NORM_EPS: Final = 1.0e-5
_FLOAT_DTYPE: Final = torch.float32
_INTEGER_DTYPE: Final = torch.int64

_Shape: TypeAlias = tuple[int, ...]
_StateSchema: TypeAlias = dict[str, tuple[_Shape, torch.dtype]]
StateDict: TypeAlias = Mapping[str, Tensor]


class MrigankaENNContractError(ValueError):
    """A checkpoint or runtime tensor violates the recovered ENN contract."""

    def __init__(self, *, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message


def _build_encoder_schema() -> _StateSchema:
    schema: _StateSchema = {
        "block1.0.mask": ((1, 1, 64, 64), _FLOAT_DTYPE),
        "block1.1.weights": ((1296,), _FLOAT_DTYPE),
        "block1.1.filter": ((192, 3, 7, 7), _FLOAT_DTYPE),
        "block1.1._basisexpansion.block_expansion_('irrep_0,0', 'regular').sampled_basis": (
            (18, 8, 1, 49),
            _FLOAT_DTYPE,
        ),
        "pool1.0.filter": ((384, 1, 5, 5), _FLOAT_DTYPE),
        "pool2.0.filter": ((768, 1, 5, 5), _FLOAT_DTYPE),
        "pool3.filter": ((512, 1, 5, 5), _FLOAT_DTYPE),
        "gpool.in_indices_8": ((2,), _INTEGER_DTYPE),
        "gpool.out_indices_8": ((2,), _INTEGER_DTYPE),
        "fully_net.0.weight": ((256, 5184), _FLOAT_DTYPE),
        "fully_net.0.bias": ((256,), _FLOAT_DTYPE),
        "fully_net.1.weight": ((256,), _FLOAT_DTYPE),
        "fully_net.1.bias": ((256,), _FLOAT_DTYPE),
        "fully_net.1.running_mean": ((256,), _FLOAT_DTYPE),
        "fully_net.1.running_var": ((256,), _FLOAT_DTYPE),
        "fully_net.1.num_batches_tracked": ((), _INTEGER_DTYPE),
    }

    convolution_specs = (
        # convolution key, batch-normalization key, basis weights, expanded filter,
        # number of regular-representation fields
        ("block2.0", "block2.1", (101376,), (384, 192, 5, 5), 48),
        ("block3.0", "block3.1", (202752,), (384, 384, 5, 5), 48),
        ("block4.0", "block4.1", (405504,), (768, 384, 5, 5), 96),
        ("block5.0", "block5.1", (811008,), (768, 768, 5, 5), 96),
        ("block6.0", "block6.1", (540672,), (512, 768, 5, 5), 64),
    )
    for (
        convolution_key,
        batch_norm_key,
        weight_shape,
        filter_shape,
        fields,
    ) in convolution_specs:
        schema[f"{convolution_key}.weights"] = (weight_shape, _FLOAT_DTYPE)
        schema[f"{convolution_key}.filter"] = (filter_shape, _FLOAT_DTYPE)
        schema[
            f"{convolution_key}._basisexpansion."
            "block_expansion_('regular', 'regular').sampled_basis"
        ] = ((88, 8, 8, 25), _FLOAT_DTYPE)
        schema[f"{batch_norm_key}.indices_8"] = ((2,), _INTEGER_DTYPE)
        for suffix in ("weight", "bias", "running_mean", "running_var"):
            schema[f"{batch_norm_key}.batch_norm_[8].{suffix}"] = (
                (fields,),
                _FLOAT_DTYPE,
            )
        schema[f"{batch_norm_key}.batch_norm_[8].num_batches_tracked"] = (
            (),
            _INTEGER_DTYPE,
        )

    schema["block1.2.indices_8"] = ((2,), _INTEGER_DTYPE)
    for suffix in ("weight", "bias", "running_mean", "running_var"):
        schema[f"block1.2.batch_norm_[8].{suffix}"] = ((24,), _FLOAT_DTYPE)
    schema["block1.2.batch_norm_[8].num_batches_tracked"] = (
        (),
        _INTEGER_DTYPE,
    )
    return schema


_ENCODER_STATE_SCHEMA: Final[_StateSchema] = _build_encoder_schema()
_CLASSIFIER_STATE_SCHEMA: Final[_StateSchema] = {
    "fc2.weight": ((2, 256), _FLOAT_DTYPE),
    "fc2.bias": ((2,), _FLOAT_DTYPE),
}

_EXPECTED_INDEX_BUFFERS: Final[dict[str, tuple[int, int]]] = {
    "block1.2.indices_8": (0, 192),
    "block2.1.indices_8": (0, 384),
    "block3.1.indices_8": (0, 384),
    "block4.1.indices_8": (0, 768),
    "block5.1.indices_8": (0, 768),
    "block6.1.indices_8": (0, 512),
    "gpool.in_indices_8": (0, 512),
    "gpool.out_indices_8": (0, 64),
}


def expected_encoder_state_shapes() -> dict[str, _Shape]:
    """Return a copy of the exact recovered encoder key-to-shape contract."""

    return {key: spec[0] for key, spec in _ENCODER_STATE_SCHEMA.items()}


def expected_classifier_state_shapes() -> dict[str, _Shape]:
    """Return a copy of the exact recovered classifier key-to-shape contract."""

    return {key: spec[0] for key, spec in _CLASSIFIER_STATE_SCHEMA.items()}


def _key_summary(keys: set[str]) -> str:
    ordered = sorted(keys)
    visible = ordered[:8]
    suffix = "" if len(ordered) <= len(visible) else f", ... ({len(ordered)} total)"
    return ", ".join(visible) + suffix


def _validate_state_dict(
    *,
    component: str,
    state_dict: StateDict,
    schema: _StateSchema,
) -> None:
    if not isinstance(state_dict, Mapping):
        raise MrigankaENNContractError(
            code=f"{component}-state-not-mapping",
            message=f"The {component} state must be a mapping of tensor names to tensors.",
        )

    observed_keys = set(state_dict.keys())
    expected_keys = set(schema)
    if any(not isinstance(key, str) for key in observed_keys):
        raise MrigankaENNContractError(
            code=f"{component}-state-key-type-invalid",
            message=f"The {component} state contains a non-string key.",
        )
    missing = expected_keys - observed_keys
    unexpected = observed_keys - expected_keys
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing: {_key_summary(missing)}")
        if unexpected:
            details.append(f"unexpected: {_key_summary(unexpected)}")
        raise MrigankaENNContractError(
            code=f"{component}-state-keys-mismatch",
            message=(
                f"The {component} checkpoint keys do not match the recovered architecture "
                f"({'; '.join(details)})."
            ),
        )

    devices: set[torch.device] = set()
    for key, (expected_shape, expected_dtype) in schema.items():
        value = state_dict[key]
        if not isinstance(value, Tensor):
            raise MrigankaENNContractError(
                code=f"{component}-state-value-not-tensor",
                message=f"The {component} checkpoint entry {key!r} is not a tensor.",
            )
        if value.layout != torch.strided or value.device.type == "meta":
            raise MrigankaENNContractError(
                code=f"{component}-state-storage-unsupported",
                message=(
                    f"The {component} checkpoint entry {key!r} must be a dense, "
                    "materialized tensor."
                ),
            )
        if tuple(value.shape) != expected_shape:
            raise MrigankaENNContractError(
                code=f"{component}-state-shape-mismatch",
                message=(
                    f"The {component} checkpoint entry {key!r} has shape "
                    f"{tuple(value.shape)!r}; expected {expected_shape!r}."
                ),
            )
        if value.dtype != expected_dtype:
            raise MrigankaENNContractError(
                code=f"{component}-state-dtype-mismatch",
                message=(
                    f"The {component} checkpoint entry {key!r} has dtype "
                    f"{value.dtype}; expected {expected_dtype}."
                ),
            )
        if expected_dtype.is_floating_point and not bool(
            torch.isfinite(value).all().item()
        ):
            raise MrigankaENNContractError(
                code=f"{component}-state-nonfinite",
                message=f"The {component} checkpoint entry {key!r} contains a non-finite value.",
            )
        devices.add(value.device)

    if len(devices) != 1:
        raise MrigankaENNContractError(
            code=f"{component}-state-device-mismatch",
            message=f"All {component} checkpoint tensors must be on one device.",
        )


def validate_encoder_state_dict(state_dict: StateDict) -> None:
    """Reject encoder states that differ from the recovered checkpoint schema."""

    _validate_state_dict(
        component="encoder",
        state_dict=state_dict,
        schema=_ENCODER_STATE_SCHEMA,
    )
    for key, expected_values in _EXPECTED_INDEX_BUFFERS.items():
        value = state_dict[key]
        expected = torch.tensor(expected_values, dtype=value.dtype, device=value.device)
        if not torch.equal(value, expected):
            raise MrigankaENNContractError(
                code="encoder-index-buffer-mismatch",
                message=(
                    f"The encoder checkpoint entry {key!r} does not describe the "
                    "contiguous regular-representation layout required by this runtime."
                ),
            )

    for key, value in state_dict.items():
        if key.endswith("running_var") and bool((value < 0).any().item()):
            raise MrigankaENNContractError(
                code="encoder-running-variance-invalid",
                message=f"The encoder checkpoint entry {key!r} contains a negative variance.",
            )
        if key.endswith("num_batches_tracked") and int(value.item()) < 0:
            raise MrigankaENNContractError(
                code="encoder-batch-count-invalid",
                message=f"The encoder checkpoint entry {key!r} contains a negative batch count.",
            )


def validate_classifier_state_dict(state_dict: StateDict) -> None:
    """Reject classifier states that differ from the recovered checkpoint schema."""

    _validate_state_dict(
        component="classifier",
        state_dict=state_dict,
        schema=_CLASSIFIER_STATE_SCHEMA,
    )


def _owned_buffer(state_dict: StateDict, key: str) -> Tensor:
    return state_dict[key].detach().clone(memory_format=torch.contiguous_format)


class _ExpandedConvBlock(nn.Module):
    """One fixed expanded group convolution followed by inner BN and ReLU."""

    def __init__(
        self,
        *,
        state_dict: StateDict,
        convolution_key: str,
        batch_norm_key: str,
        padding: int,
        fields: int,
    ) -> None:
        super().__init__()
        self.padding = padding
        self.fields = fields
        self.register_buffer(
            "expanded_filter",
            _owned_buffer(state_dict, f"{convolution_key}.filter"),
        )
        self.register_buffer(
            "batch_norm_weight",
            _owned_buffer(state_dict, f"{batch_norm_key}.batch_norm_[8].weight"),
        )
        self.register_buffer(
            "batch_norm_bias",
            _owned_buffer(state_dict, f"{batch_norm_key}.batch_norm_[8].bias"),
        )
        self.register_buffer(
            "running_mean",
            _owned_buffer(state_dict, f"{batch_norm_key}.batch_norm_[8].running_mean"),
        )
        self.register_buffer(
            "running_var",
            _owned_buffer(state_dict, f"{batch_norm_key}.batch_norm_[8].running_var"),
        )

    def forward(self, image: Tensor) -> Tensor:
        output = F.conv2d(image, self.expanded_filter, padding=self.padding)
        batch, channels, height, width = output.shape
        expected_channels = self.fields * REGULAR_REPRESENTATION_SIZE
        if channels != expected_channels:
            raise RuntimeError(
                "The translated ENN convolution produced an unexpected channel count: "
                f"{channels}; expected {expected_channels}."
            )
        output = output.reshape(
            batch,
            self.fields,
            REGULAR_REPRESENTATION_SIZE,
            height,
            width,
        )
        output = F.batch_norm(
            output,
            self.running_mean,
            self.running_var,
            self.batch_norm_weight,
            self.batch_norm_bias,
            training=False,
            momentum=0.1,
            eps=_BATCH_NORM_EPS,
        )
        return F.relu(output.reshape(batch, channels, height, width), inplace=False)


class MrigankaENNEncoder(nn.Module):
    """Fixed evaluation-mode encoder translated from one validated state dict."""

    def __init__(self, state_dict: StateDict) -> None:
        super().__init__()
        validate_encoder_state_dict(state_dict)

        self.register_buffer("input_mask", _owned_buffer(state_dict, "block1.0.mask"))
        self.block1 = _ExpandedConvBlock(
            state_dict=state_dict,
            convolution_key="block1.1",
            batch_norm_key="block1.2",
            padding=1,
            fields=24,
        )
        self.block2 = _ExpandedConvBlock(
            state_dict=state_dict,
            convolution_key="block2.0",
            batch_norm_key="block2.1",
            padding=2,
            fields=48,
        )
        self.block3 = _ExpandedConvBlock(
            state_dict=state_dict,
            convolution_key="block3.0",
            batch_norm_key="block3.1",
            padding=2,
            fields=48,
        )
        self.block4 = _ExpandedConvBlock(
            state_dict=state_dict,
            convolution_key="block4.0",
            batch_norm_key="block4.1",
            padding=2,
            fields=96,
        )
        self.block5 = _ExpandedConvBlock(
            state_dict=state_dict,
            convolution_key="block5.0",
            batch_norm_key="block5.1",
            padding=2,
            fields=96,
        )
        self.block6 = _ExpandedConvBlock(
            state_dict=state_dict,
            convolution_key="block6.0",
            batch_norm_key="block6.1",
            padding=1,
            fields=64,
        )

        self.register_buffer(
            "pool1_filter", _owned_buffer(state_dict, "pool1.0.filter")
        )
        self.register_buffer(
            "pool2_filter", _owned_buffer(state_dict, "pool2.0.filter")
        )
        self.register_buffer("pool3_filter", _owned_buffer(state_dict, "pool3.filter"))
        self.register_buffer(
            "projection_weight",
            _owned_buffer(state_dict, "fully_net.0.weight"),
        )
        self.register_buffer(
            "projection_bias",
            _owned_buffer(state_dict, "fully_net.0.bias"),
        )
        self.register_buffer(
            "projection_batch_norm_weight",
            _owned_buffer(state_dict, "fully_net.1.weight"),
        )
        self.register_buffer(
            "projection_batch_norm_bias",
            _owned_buffer(state_dict, "fully_net.1.bias"),
        )
        self.register_buffer(
            "projection_running_mean",
            _owned_buffer(state_dict, "fully_net.1.running_mean"),
        )
        self.register_buffer(
            "projection_running_var",
            _owned_buffer(state_dict, "fully_net.1.running_var"),
        )
        self.eval()

    @property
    def device(self) -> torch.device:
        return self.input_mask.device

    def _validate_input(self, image: Tensor) -> None:
        if not isinstance(image, Tensor):
            raise MrigankaENNContractError(
                code="encoder-input-not-tensor",
                message="The ENN encoder input must be a PyTorch tensor.",
            )
        if image.layout != torch.strided or image.device.type == "meta":
            raise MrigankaENNContractError(
                code="encoder-input-storage-unsupported",
                message="The ENN encoder input must be a dense, materialized tensor.",
            )
        if image.ndim != 4 or tuple(image.shape[1:]) != (
            INPUT_CHANNELS,
            INPUT_HEIGHT,
            INPUT_WIDTH,
        ):
            raise MrigankaENNContractError(
                code="encoder-input-shape-mismatch",
                message=(
                    "The ENN encoder input must have shape [batch, 3, 64, 64]; "
                    f"received {tuple(image.shape)!r}."
                ),
            )
        if image.shape[0] < 1:
            raise MrigankaENNContractError(
                code="encoder-input-empty-batch",
                message="The ENN encoder input batch must contain at least one image.",
            )
        if image.dtype != torch.float32:
            raise MrigankaENNContractError(
                code="encoder-input-dtype-mismatch",
                message=f"The ENN encoder input must be float32; received {image.dtype}.",
            )
        if self.input_mask.dtype != torch.float32:
            raise MrigankaENNContractError(
                code="encoder-runtime-dtype-mismatch",
                message="The ENN runtime buffers must remain float32.",
            )
        if image.device != self.device:
            raise MrigankaENNContractError(
                code="encoder-input-device-mismatch",
                message=(
                    f"The ENN input is on {image.device}, but the encoder is on {self.device}."
                ),
            )
        if not bool(torch.isfinite(image).all().item()):
            raise MrigankaENNContractError(
                code="encoder-input-nonfinite",
                message="The ENN encoder input contains a non-finite value.",
            )
        if bool((image < 0.0).any().item()) or bool((image > 1.0).any().item()):
            raise MrigankaENNContractError(
                code="encoder-input-range-mismatch",
                message="The ENN encoder input values must lie in the closed interval [0, 1].",
            )

    @staticmethod
    def _antialiased_pool(
        image: Tensor,
        fixed_filter: Tensor,
        *,
        stride: int,
        padding: int,
    ) -> Tensor:
        return F.conv2d(
            image,
            fixed_filter,
            stride=stride,
            padding=padding,
            groups=image.shape[1],
        )

    def forward(self, image: Tensor) -> Tensor:
        self._validate_input(image)
        output = self.block1(image * self.input_mask)
        output = self.block2(output)
        output = self._antialiased_pool(
            output,
            self.pool1_filter,
            stride=2,
            padding=2,
        )
        output = self.block3(output)
        output = self.block4(output)
        output = self._antialiased_pool(
            output,
            self.pool2_filter,
            stride=2,
            padding=2,
        )
        output = self.block5(output)
        output = self.block6(output)
        output = self._antialiased_pool(
            output,
            self.pool3_filter,
            stride=1,
            padding=0,
        )

        batch, channels, height, width = output.shape
        if (channels, height, width) != (512, 9, 9):
            raise RuntimeError(
                "The translated ENN encoder produced an unexpected feature shape: "
                f"{(channels, height, width)!r}; expected (512, 9, 9)."
            )
        output = output.reshape(
            batch,
            64,
            REGULAR_REPRESENTATION_SIZE,
            height,
            width,
        ).amax(dim=2)
        output = F.linear(
            output.reshape(batch, 5184),
            self.projection_weight,
            self.projection_bias,
        )
        return F.batch_norm(
            output,
            self.projection_running_mean,
            self.projection_running_var,
            self.projection_batch_norm_weight,
            self.projection_batch_norm_bias,
            training=False,
            momentum=0.1,
            eps=_BATCH_NORM_EPS,
        )


class MrigankaENNClassifier(nn.Module):
    """Two-logit lens classifier head recovered from the supplied checkpoint."""

    def __init__(self, state_dict: StateDict) -> None:
        super().__init__()
        validate_classifier_state_dict(state_dict)
        self.register_buffer("weight", _owned_buffer(state_dict, "fc2.weight"))
        self.register_buffer("bias", _owned_buffer(state_dict, "fc2.bias"))
        self.eval()

    @property
    def device(self) -> torch.device:
        return self.weight.device

    def forward(self, features: Tensor) -> Tensor:
        if not isinstance(features, Tensor):
            raise MrigankaENNContractError(
                code="classifier-input-not-tensor",
                message="The ENN classifier input must be a PyTorch tensor.",
            )
        if features.layout != torch.strided or features.device.type == "meta":
            raise MrigankaENNContractError(
                code="classifier-input-storage-unsupported",
                message="The ENN classifier input must be a dense, materialized tensor.",
            )
        if (
            features.ndim != 2
            or features.shape[0] < 1
            or features.shape[1] != LATENT_SIZE
        ):
            raise MrigankaENNContractError(
                code="classifier-input-shape-mismatch",
                message=(
                    "The ENN classifier input must have shape [batch, 256]; "
                    f"received {tuple(features.shape)!r}."
                ),
            )
        if features.dtype != torch.float32 or self.weight.dtype != torch.float32:
            raise MrigankaENNContractError(
                code="classifier-input-dtype-mismatch",
                message="The ENN classifier input and runtime buffers must be float32.",
            )
        if features.device != self.device:
            raise MrigankaENNContractError(
                code="classifier-input-device-mismatch",
                message=(
                    f"The ENN features are on {features.device}, but the classifier is on "
                    f"{self.device}."
                ),
            )
        if not bool(torch.isfinite(features).all().item()):
            raise MrigankaENNContractError(
                code="classifier-input-nonfinite",
                message="The ENN classifier input contains a non-finite value.",
            )
        return F.linear(F.relu(features, inplace=False), self.weight, self.bias)


class MrigankaENN(nn.Module):
    """Combined encoder and two-logit classifier for deterministic inference."""

    def __init__(self, encoder_state: StateDict, classifier_state: StateDict) -> None:
        super().__init__()
        self.encoder = MrigankaENNEncoder(encoder_state)
        self.classifier = MrigankaENNClassifier(classifier_state)
        if self.encoder.device != self.classifier.device:
            raise MrigankaENNContractError(
                code="checkpoint-device-mismatch",
                message="The encoder and classifier checkpoint tensors must be on one device.",
            )
        self.eval()

    def encode(self, image: Tensor) -> Tensor:
        """Return the raw 256-value encoder output before classifier ReLU."""

        return self.encoder(image)

    def forward(self, image: Tensor) -> Tensor:
        """Return two raw logits; probability conversion belongs to the caller."""

        return self.classifier(self.encode(image))


__all__ = [
    "INPUT_CHANNELS",
    "INPUT_HEIGHT",
    "INPUT_WIDTH",
    "LATENT_SIZE",
    "LOGIT_COUNT",
    "MrigankaENN",
    "MrigankaENNClassifier",
    "MrigankaENNContractError",
    "MrigankaENNEncoder",
    "expected_classifier_state_shapes",
    "expected_encoder_state_shapes",
    "validate_classifier_state_dict",
    "validate_encoder_state_dict",
]
