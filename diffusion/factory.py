"""Build a noise-prediction network from its architecture configuration."""
from .advanced import QualityUNet
from .network import NoiseUNet


def create_model(config):
    options = dict(config)
    architecture = options.pop("architecture", "baseline")
    if architecture == "baseline":
        return NoiseUNet(**options)
    if architecture == "quality_unet":
        return QualityUNet(**options)
    raise ValueError(f"Unknown architecture: {architecture}")
