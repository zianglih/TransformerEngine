# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from transformer_engine.pytorch.module._common import apply_nvfp4_per_token_activation_scaling

nvfp4_available, reason_for_no_nvfp4 = te.is_nvfp4_available(return_reason=True)
bf16_available, reason_for_no_bf16 = te.is_bf16_available(return_reason=True)


@pytest.mark.skipif(not nvfp4_available, reason=reason_for_no_nvfp4)
def test_nvfp4_recipe_per_token_activation_flag():
    default_recipe = recipe.NVFP4BlockScaling()
    assert not default_recipe.per_token_activation

    per_token_recipe = recipe.NVFP4BlockScaling(per_token_activation=True)
    assert per_token_recipe.per_token_activation


@pytest.mark.skipif(not nvfp4_available, reason=reason_for_no_nvfp4)
@pytest.mark.skipif(not bf16_available, reason=reason_for_no_bf16)
def test_grouped_linear_per_token_activation_matches_manual_rescale():
    torch.manual_seed(1234)
    device = "cuda"
    dtype = torch.bfloat16
    num_groups = 2
    in_features = 64
    out_features = 64
    m_splits = [64, 64]

    module = te.GroupedLinear(
        num_gemms=num_groups,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        params_dtype=dtype,
        device=device,
    )
    module.eval()

    x = torch.randn(sum(m_splits), in_features, device=device, dtype=dtype)
    # Build pronounced per-row dynamic range variation inside each split.
    x[:63] *= 1.0e-3
    x[63] *= 1.0e2
    x[64:127] *= 2.0e-3
    x[127] *= 8.0e1

    base_recipe = recipe.NVFP4BlockScaling(
        disable_rht=True,
        disable_stochastic_rounding=True,
        disable_2d_quantization=True,
        per_token_activation=False,
    )
    per_token_recipe = recipe.NVFP4BlockScaling(
        disable_rht=True,
        disable_stochastic_rounding=True,
        disable_2d_quantization=True,
        per_token_activation=True,
    )

    x_scaled, output_rescale = apply_nvfp4_per_token_activation_scaling(x, m_splits)
    with torch.no_grad():
        with te.autocast(enabled=True, recipe=base_recipe):
            y_manual = module(x_scaled, m_splits)
        y_manual = y_manual.reshape(-1, y_manual.shape[-1])
        y_manual.mul_(output_rescale.to(dtype=y_manual.dtype).unsqueeze(-1))
        y_manual = y_manual.reshape(sum(m_splits), out_features)

        with te.autocast(enabled=True, recipe=per_token_recipe):
            y_recipe = module(x, m_splits)
        y_recipe = y_recipe.reshape(sum(m_splits), out_features)

    torch.testing.assert_close(y_recipe, y_manual, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not nvfp4_available, reason=reason_for_no_nvfp4)
@pytest.mark.skipif(not bf16_available, reason=reason_for_no_bf16)
def test_grouped_linear_per_token_activation_bias_backward_dequantized():
    torch.manual_seed(4321)
    device = "cuda"
    dtype = torch.bfloat16
    m_splits = [64, 64]

    module = te.GroupedLinear(
        num_gemms=2,
        in_features=64,
        out_features=64,
        bias=True,
        params_dtype=dtype,
        device=device,
    )
    module.train()

    x = torch.randn(sum(m_splits), 64, device=device, dtype=dtype)
    x[:63] *= 1.0e-3
    x[63] *= 1.0e2
    x[64:127] *= 2.0e-3
    x[127] *= 8.0e1
    x.requires_grad_(True)

    per_token_recipe = recipe.NVFP4BlockScaling(
        disable_rht=True,
        disable_stochastic_rounding=True,
        disable_2d_quantization=True,
        per_token_activation=True,
        backward_override="dequantized",
    )

    with te.autocast(enabled=True, recipe=per_token_recipe):
        y = module(x, m_splits)
        loss = y.float().pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert module.weight0.grad is not None
    assert torch.isfinite(module.weight0.grad).all()
    assert module.bias0.grad is not None
    assert torch.isfinite(module.bias0.grad).all()
