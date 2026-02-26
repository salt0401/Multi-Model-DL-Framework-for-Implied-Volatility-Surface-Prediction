"""Shared pytest fixtures for model1_research tests."""
import pytest
import torch

torch.set_default_dtype(torch.float64)


@pytest.fixture
def tiny_batch():
    """Generate a small batch of synthetic data (tau, logm, yATM, y_true)."""
    n = 16
    tau = torch.rand(n, 1) * 1.98 + 0.02       # tau in [0.02, 2.0]
    logm = torch.rand(n, 1) * 1.0 - 0.5         # logm in [-0.5, 0.5]
    yATM = torch.rand(n, 1) * 0.099 + 0.001     # yATM in [0.001, 0.1]
    y_true = torch.rand(n, 1) * 0.05 + 0.001    # total variance target
    return tau, logm, yATM, y_true
