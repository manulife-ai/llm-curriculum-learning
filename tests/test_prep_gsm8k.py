import json

from src.data.prep.gsm8k import prepare_gsm8k, split_train_val


def _fake_loader(*, train_rows, test_rows):
    def loader(dataset_id, config_name, split):
        assert dataset_id == "openai/gsm8k"
        assert config_name == "main"
        return train_rows if split == "train" else test_rows
    return loader


def test_split_train_val_is_deterministic():
    rows = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(20)]
    a_train, a_val = split_train_val(rows, val_size=5, seed=123)
    b_train, b_val = split_train_val(rows, val_size=5, seed=123)
    assert a_train == b_train
    assert a_val == b_val
    assert len(a_val) == 5
    assert len(a_train) == 15
    # Union of indices covers all rows exactly once.
    assert {r["question"] for r in a_train} | {r["question"] for r in a_val} == {r["question"] for r in rows}


def test_split_train_val_differs_across_seeds():
    rows = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(30)]
    _, val1 = split_train_val(rows, val_size=8, seed=1)
    _, val2 = split_train_val(rows, val_size=8, seed=2)
    assert val1 != val2


def test_prepare_gsm8k_writes_expected_files_and_manifest(tmp_path):
    train_rows = [{"question": f"q{i}", "answer": f"a{i} #### {i}"} for i in range(10)]
    test_rows = [{"question": f"tq{i}", "answer": f"ta{i} #### {i}"} for i in range(4)]
    manifest = prepare_gsm8k(
        output_dir=tmp_path,
        manifest_path=tmp_path / "manifests" / "gsm8k.json",
        val_size=2,
        seed=7,
        dataset_loader=_fake_loader(train_rows=train_rows, test_rows=test_rows),
    )
    assert manifest["counts"] == {"train": 8, "val": 2, "test": 4}
    for split in ("train", "val", "test"):
        path = tmp_path / f"gsm8k_{split}.jsonl"
        assert path.exists()
        with path.open() as handle:
            records = [json.loads(line) for line in handle]
        assert len(records) == manifest["counts"][split]
        for r in records:
            assert set(r.keys()) == {"id", "input_text", "target_text"}
            assert r["id"].startswith(f"gsm8k_{split}_")
    # Manifest hashes match freshly computed hashes.
    manifest_disk = json.loads((tmp_path / "manifests" / "gsm8k.json").read_text())
    assert manifest_disk == manifest


def test_prepare_gsm8k_is_reproducible(tmp_path):
    train_rows = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(12)]
    test_rows = [{"question": f"tq{i}", "answer": f"ta{i}"} for i in range(3)]
    kwargs = dict(
        val_size=3,
        seed=99,
        dataset_loader=_fake_loader(train_rows=train_rows, test_rows=test_rows),
    )
    m1 = prepare_gsm8k(output_dir=tmp_path / "a", manifest_path=tmp_path / "a" / "m.json", **kwargs)
    m2 = prepare_gsm8k(output_dir=tmp_path / "b", manifest_path=tmp_path / "b" / "m.json", **kwargs)
    assert m1["sha256"] == m2["sha256"]
