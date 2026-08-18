import random

import numpy as np
import torch

from src.utils.seed import build_generator, set_seed


def test_set_seed_makes_stdlib_random_deterministic():
    set_seed(123)
    a = [random.random() for _ in range(5)]
    set_seed(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_seed_makes_numpy_deterministic():
    set_seed(7)
    a = np.random.rand(4)
    set_seed(7)
    b = np.random.rand(4)
    assert np.array_equal(a, b)


def test_set_seed_makes_torch_deterministic():
    set_seed(42)
    a = torch.rand(4)
    set_seed(42)
    b = torch.rand(4)
    assert torch.equal(a, b)


def test_build_generator_is_deterministic():
    g1 = build_generator(99)
    g2 = build_generator(99)
    a = torch.rand(4, generator=g1)
    b = torch.rand(4, generator=g2)
    assert torch.equal(a, b)


def test_build_generator_differs_across_seeds():
    a = torch.rand(4, generator=build_generator(1))
    b = torch.rand(4, generator=build_generator(2))
    assert not torch.equal(a, b)
