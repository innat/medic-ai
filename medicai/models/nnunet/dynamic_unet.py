from .unet import UNet
from medicai.trainer.nnunet.utils.config import NetworkConfig, nnUNetPlan


def build_unet_from_plan(
    plan,
    configuration="3d_fullres",
):
    """
    Build a UNet from a saved plan.

    Parameters
    ----------
    plan          : nnUNetPlan loaded from ``nnunet_plans.json``
    configuration : which sub-plan to use: '3d_fullres', '3d_lowres', or '2d'

    Returns
    -------
    UNet (not yet compiled; call model.compile() before training)
    """
    cfg_map = {
        "3d_fullres": plan.plan_3d_fullres,
        "3d_lowres": plan.plan_3d_lowres,
        "2d": plan.plan_2d,
    }
    net_cfg = cfg_map.get(configuration)
    if net_cfg is None:
        raise ValueError(
            f"Configuration '{configuration}' not found in plan. "
            f"Available: {[k for k, v in cfg_map.items() if v is not None]}"
        )

    return build_unet_from_config(net_cfg)


def build_unet_from_config(cfg):
    """
    Build a UNet from a NetworkConfig.

    This is the low-level factory used by ``build_unet_from_plan`` and by
    unit tests.
    """
    # Derive conv kernel sizes from pool op kernels
    # Use [3,3,3] for in-plane dims and [1,3,3] where pool stride is 1
    # (anisotropic handling)
    conv_kernel_sizes = _derive_conv_kernels(
        cfg.pool_op_kernel_sizes,
        cfg.spatial_dims,
        cfg.n_pooling,
    )

    return UNet(
        spatial_dims=cfg.spatial_dims,
        n_classes=cfg.n_classes,
        n_input_channels=cfg.n_modalities,
        n_pooling=cfg.n_pooling,
        base_filters=cfg.base_filters,
        max_filters=cfg.max_filters,
        pool_op_kernel_sizes=cfg.pool_op_kernel_sizes if cfg.pool_op_kernel_sizes else None,
        conv_kernel_sizes=conv_kernel_sizes,
        deep_supervision=cfg.deep_supervision,
        output_activation=cfg.output_activation,
    )


def _derive_conv_kernels(
    pool_op_kernel_sizes,
    spatial_dims,
    n_pooling,
):
    """
    Derive convolution kernel sizes from the pool op kernel sizes.

    Rule: where pool stride is 1 (anisotropic, thick axis), use kernel=1
    in that axis (avoids blurring across slice). Otherwise use kernel=3.
    Applied to each stage (n_pooling + 1 for the bottleneck).
    """
    if not pool_op_kernel_sizes:
        k = [3] * spatial_dims
        return [k] * (n_pooling + 1)

    conv_kernels = []
    for pool_k in pool_op_kernel_sizes:
        # kernel = 3 where we pool (stride=2), kernel = 1 where we don't
        ck = [3 if p >= 2 else 1 for p in pool_k]
        conv_kernels.append(ck)

    # Add bottleneck (use last stage kernel)
    conv_kernels.append(conv_kernels[-1] if conv_kernels else [3] * spatial_dims)

    return conv_kernels
