import torch
from src.methods.compression.quantile import QuantileTrimmer


def test_quantile_shrinkage():
    torch.testing.assert_close(
        logits_processor_output[:, 1],
        fake_logits[:, 1],
        rtol=1e-4,
        atol=0.0
    )
    QuantileTrimmer()
