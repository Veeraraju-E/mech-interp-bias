"""Linear probing analysis: train probes on activations to predict bias."""

from .probe import train_layer_probes, ProbeResults

__all__ = ['train_layer_probes', 'ProbeResults']

