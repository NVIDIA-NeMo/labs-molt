from .fp8 import (
    HAVE_TORCHAO,
    FP8Config,
    apply_fp8_to_model,
    build_fp8_config,
    create_fp8_config_from_dict,
    verify_fp8_conversion,
)
if HAVE_TORCHAO:
    from torchao.float8 import Float8LinearConfig
else:
    Float8LinearConfig = None

__all__ = [
    "apply_fp8_to_model",
    "verify_fp8_conversion",
    "build_fp8_config",
    "create_fp8_config_from_dict",
    "HAVE_TORCHAO",
    "FP8Config",
]

if HAVE_TORCHAO:
    __all__.append("Float8LinearConfig")
