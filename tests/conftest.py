import pytest

# add optinions to control experiments
def pytest_addoption(parser):
    parser.addoption(
        "--num_test_samples", action="store", default="10", type=int
    )
    parser.addoption(
        "--model_dim", action="store", default="5", type=int
    )