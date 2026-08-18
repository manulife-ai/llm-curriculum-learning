"""Tests for the Δ(k) exposure-bias diagnostic using a toy model."""

import pytest

torch = pytest.importorskip("torch")

from src.eval.exposure_bias import compute_delta_k


class ToyTokenizer:
    """Whitespace tokenizer that maps single-character words to their index."""

    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Each word is a single letter/digit; ord() gives a small stable id.
        return [ord(tok) for tok in text.split()]


class ToyModel:
    """Toy model whose forward loss depends on whether an 'own' marker appears in the prefix.

    - `.generate()` always emits a fixed continuation.
    - `.__call__()` returns loss=`nll_when_own_present` if any own-marker token appears
      in the prefix region of input_ids, otherwise `nll_when_gold`.

    This lets us verify that Δ(k) is exactly `nll_when_own_present - nll_when_gold`
    without training a real model.
    """

    def __init__(
        self,
        own_generation_ids: list[int],
        own_marker_id: int,
        nll_when_own_present: float,
        nll_when_gold: float,
    ):
        self.own_generation_ids = own_generation_ids
        self.own_marker_id = own_marker_id
        self.nll_when_own_present = nll_when_own_present
        self.nll_when_gold = nll_when_gold

    def eval(self):
        return self

    def generate(self, input_ids, attention_mask, max_new_tokens, do_sample, num_beams, pad_token_id):
        continuation = torch.tensor(
            self.own_generation_ids[:max_new_tokens], dtype=torch.long
        ).unsqueeze(0)
        return torch.cat([input_ids.cpu(), continuation], dim=1)

    def __call__(self, input_ids, labels):
        # The prefix region is everything before the first non-IGNORE label.
        non_ignore = (labels != -100).nonzero(as_tuple=True)
        if non_ignore[0].numel() == 0:
            loss_value = self.nll_when_gold
        else:
            continuation_start = int(non_ignore[1].min().item())
            prefix_tokens = input_ids[0, :continuation_start].tolist()
            has_own = self.own_marker_id in prefix_tokens
            loss_value = self.nll_when_own_present if has_own else self.nll_when_gold
        return type("Output", (), {"loss": torch.tensor(loss_value)})()


def test_delta_k_returns_expected_structure():
    tokenizer = ToyTokenizer()
    model = ToyModel(
        own_generation_ids=[ord("z"), ord("z"), ord("z"), ord("z")],
        own_marker_id=ord("z"),
        nll_when_own_present=1.2,
        nll_when_gold=0.4,
    )
    examples = [
        {"prompt_text": "p q", "target_text": "a b c d e"},
        {"prompt_text": "r", "target_text": "a b c d e"},
    ]
    result = compute_delta_k(model, tokenizer, examples, k_values=[1, 2], device="cpu")
    assert set(result.keys()) == {"k_values", "nll_own", "nll_gold", "delta_k", "counts"}
    assert result["k_values"] == [1, 2]
    assert set(result["delta_k"].keys()) == {1, 2}


def test_delta_k_matches_analytical_gap_when_own_prefix_differs_from_gold():
    tokenizer = ToyTokenizer()
    model = ToyModel(
        own_generation_ids=[ord("z"), ord("z"), ord("z"), ord("z")],
        own_marker_id=ord("z"),
        nll_when_own_present=1.2,
        nll_when_gold=0.4,
    )
    examples = [{"prompt_text": "p", "target_text": "a b c d e"}] * 3
    result = compute_delta_k(model, tokenizer, examples, k_values=[1, 2], device="cpu")
    for k in (1, 2):
        assert result["nll_own"][k] == pytest.approx(1.2)
        assert result["nll_gold"][k] == pytest.approx(0.4)
        assert result["delta_k"][k] == pytest.approx(0.8)
        assert result["counts"][k] == 3


def test_delta_k_is_zero_when_own_and_gold_prefixes_are_identical():
    tokenizer = ToyTokenizer()
    # Own generation matches the gold prefix, so the marker never appears — both losses equal.
    model = ToyModel(
        own_generation_ids=[ord("a"), ord("b"), ord("c"), ord("d")],
        own_marker_id=ord("z"),
        nll_when_own_present=1.2,
        nll_when_gold=0.4,
    )
    examples = [{"prompt_text": "p", "target_text": "a b c d e"}]
    result = compute_delta_k(model, tokenizer, examples, k_values=[1, 2], device="cpu")
    for k in (1, 2):
        assert result["delta_k"][k] == pytest.approx(0.0)


def test_delta_k_skips_examples_shorter_than_k():
    tokenizer = ToyTokenizer()
    model = ToyModel(
        own_generation_ids=[ord("z")] * 8,
        own_marker_id=ord("z"),
        nll_when_own_present=1.0,
        nll_when_gold=0.5,
    )
    examples = [
        {"prompt_text": "p", "target_text": "a b"},  # only 2 tokens, k=4 must skip
        {"prompt_text": "p", "target_text": "a b c d e f"},  # eligible for k=4
    ]
    result = compute_delta_k(model, tokenizer, examples, k_values=[4], device="cpu")
    assert result["counts"][4] == 1


def test_delta_k_raises_on_empty_k_values():
    tokenizer = ToyTokenizer()
    model = ToyModel([], 0, 0.0, 0.0)
    with pytest.raises(ValueError):
        compute_delta_k(model, tokenizer, [], k_values=[], device="cpu")


def test_delta_k_max_examples_limits_iteration():
    tokenizer = ToyTokenizer()
    model = ToyModel(
        own_generation_ids=[ord("z")] * 4,
        own_marker_id=ord("z"),
        nll_when_own_present=1.0,
        nll_when_gold=0.5,
    )
    examples = [{"prompt_text": "p", "target_text": "a b c d e"} for _ in range(10)]
    result = compute_delta_k(model, tokenizer, examples, k_values=[1], device="cpu", max_examples=3)
    assert result["counts"][1] == 3
