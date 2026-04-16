import torch

from src.curriculum.mixer import SequentialGreedyRollInMixer, TokenLevelGreedyMixer, build_mixer


class DummyModel:
    def __call__(self, input_ids, attention_mask):
        batch, seq_len = input_ids.shape
        vocab_size = 8
        logits = torch.zeros(batch, seq_len, vocab_size)
        logits[..., 3] = 1.0
        return type("Output", (), {"logits": logits})


class AutoregressiveDependencyModel:
    def __call__(self, input_ids, attention_mask):
        batch, seq_len = input_ids.shape
        vocab_size = 64
        logits = torch.zeros(batch, seq_len, vocab_size)
        next_tokens = torch.clamp(input_ids + 10, max=vocab_size - 1)
        for batch_index in range(batch):
            for position in range(seq_len):
                logits[batch_index, position, int(next_tokens[batch_index, position].item())] = 1.0
        return type("Output", (), {"logits": logits})


def test_mixer_leaves_inputs_unchanged_when_probability_zero():
    mixer = TokenLevelGreedyMixer()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "labels": torch.tensor([[-100, 2, 3]]),
    }
    mixed, stats = mixer.mix(DummyModel(), batch, 0.0)
    assert torch.equal(mixed, batch["input_ids"])
    assert stats["replaced_tokens"] == 0


def test_single_pass_mixer_replaces_target_tokens_in_parallel():
    mixer = TokenLevelGreedyMixer()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        "labels": torch.tensor([[-100, 2, 3, 4]]),
    }
    mixed, stats = mixer.mix(AutoregressiveDependencyModel(), batch, 1.0)
    assert torch.equal(mixed, torch.tensor([[1, 11, 12, 13]]))
    assert stats["mixer_type"] == "single_pass"
    assert stats["replaced_tokens"] == 3


def test_sequential_roll_in_uses_updated_predictions_for_later_tokens():
    mixer = SequentialGreedyRollInMixer()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        "labels": torch.tensor([[-100, 2, 3, 4]]),
    }
    mixed, stats = mixer.mix(AutoregressiveDependencyModel(), batch, 1.0)
    assert torch.equal(mixed, torch.tensor([[1, 11, 21, 31]]))
    assert stats["mixer_type"] == "sequential"
    assert stats["replaced_tokens"] == 3


def test_build_mixer_selects_configured_strategy():
    assert isinstance(build_mixer({"curriculum": {"mixer_type": "single_pass"}}), TokenLevelGreedyMixer)
    assert isinstance(build_mixer({"curriculum": {"mixer_type": "sequential"}}), SequentialGreedyRollInMixer)
