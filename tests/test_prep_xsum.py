import json

from src.data.prep.xsum import deterministic_subsample, prepare_xsum


def _fake_loader(*, train_rows, val_rows, test_rows):
    def loader(dataset_id, split):
        assert dataset_id == "EdinburghNLP/xsum"
        if split == "train":
            return train_rows
        if split == "validation":
            return val_rows
        if split == "test":
            return test_rows
        raise AssertionError(f"unexpected split: {split}")
    return loader


def test_deterministic_subsample_is_stable():
    rows = [{"document": f"d{i}", "summary": f"s{i}"} for i in range(50)]
    a = deterministic_subsample(rows, size=10, seed=7)
    b = deterministic_subsample(rows, size=10, seed=7)
    assert a == b
    assert len(a) == 10
    assert len({r["document"] for r in a}) == 10  # no duplicates


def test_deterministic_subsample_differs_across_seeds():
    rows = [{"document": f"d{i}", "summary": f"s{i}"} for i in range(50)]
    a = deterministic_subsample(rows, size=10, seed=1)
    b = deterministic_subsample(rows, size=10, seed=2)
    assert a != b


def test_prepare_xsum_writes_expected_files_and_manifest(tmp_path):
    train = [{"document": f"td{i}", "summary": f"ts{i}", "id": str(i)} for i in range(40)]
    val = [{"document": f"vd{i}", "summary": f"vs{i}", "id": str(i)} for i in range(8)]
    test = [{"document": f"ted{i}", "summary": f"tes{i}", "id": str(i)} for i in range(10)]
    manifest = prepare_xsum(
        output_dir=tmp_path,
        manifest_path=tmp_path / "manifests" / "xsum.json",
        train_size=15,
        val_size=5,
        test_size=6,
        seed=3,
        dataset_loader=_fake_loader(train_rows=train, val_rows=val, test_rows=test),
    )
    assert manifest["counts"] == {"train": 15, "val": 5, "test": 6}
    for split, expected in [("train", 15), ("val", 5), ("test", 6)]:
        path = tmp_path / f"xsum_{split}.jsonl"
        assert path.exists()
        with path.open() as handle:
            records = [json.loads(line) for line in handle]
        assert len(records) == expected
        for r in records:
            assert set(r.keys()) == {"id", "input_text", "target_text"}
            assert r["id"].startswith(f"xsum_{split}_")
    manifest_disk = json.loads((tmp_path / "manifests" / "xsum.json").read_text())
    assert manifest_disk == manifest


def test_prepare_xsum_is_reproducible(tmp_path):
    train = [{"document": f"td{i}", "summary": f"ts{i}"} for i in range(30)]
    val = [{"document": f"vd{i}", "summary": f"vs{i}"} for i in range(10)]
    test = [{"document": f"ted{i}", "summary": f"tes{i}"} for i in range(10)]
    kwargs = dict(
        train_size=12,
        val_size=4,
        test_size=5,
        seed=11,
        dataset_loader=_fake_loader(train_rows=train, val_rows=val, test_rows=test),
    )
    m1 = prepare_xsum(output_dir=tmp_path / "a", manifest_path=tmp_path / "a" / "m.json", **kwargs)
    m2 = prepare_xsum(output_dir=tmp_path / "b", manifest_path=tmp_path / "b" / "m.json", **kwargs)
    assert m1["sha256"] == m2["sha256"]
