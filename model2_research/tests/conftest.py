import sys
import os
import torch
import pytest

# Allow direct imports from the parent directory (model2_research)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture(scope='session', autouse=True)
def set_float64():
    """Set default dtype to float64 for entire session (matches production)."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)
