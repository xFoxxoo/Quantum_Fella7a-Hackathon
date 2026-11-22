"""Submission interface on four training-fitted PCA features, without IDs."""
from .models import QuantumKernelSVM


class Submission(QuantumKernelSVM):
    def __init__(self):
        super().__init__(labels=('alfalfa', 'maize', 'wheat'), scale=0.35, C=3.0)