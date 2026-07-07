"""Job-array consolidation: merge fragments, reproduce single-node regret, export CSVs.

Unit tests drive :func:`consolidate_run` with hand-built fragment rows (registry order
faked via ``load_specs`` monkeypatching, pattern from ``test_full_eval_smoke``); the
end-to-end test proves the acceptance criterion — single-node ``run_full_eval`` vs
per-model task runs + consolidation produce byte-identical ``regret.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from mlsys.search.consolidate import consolidate_run, flatten_row

_TIMING = {
    "prepare_model_s": 0.1,
    "prepare_data_s": 0.1,
    "inference_s": 0.1,
    "train_head_s": 0.1,
    "eval_s": 0.1,
    "peak_gpu_mem_mb": 0.0,
}


def _row(model: str, strategy: str, r2: float, dataset: str = "synthetic", **extras) -> dict:
    return {
        "dataset": dataset,
        "model": model,
        "strategy": strategy,
        "metrics": {"mse": 1.0 - r2, "mae": 0.5, "r2": r2, "spearman": 0.9},
        "timing": dict(_TIMING),
        "head_train_curve": [1.0, 0.5],
        "head_val_curve": [1.0, 0.6],
        "epochs_run": 2,
        "embedding_dim": 8,
        "head_type": "linear",
        "head_repeats": 3,
        **extras,
    }


def _write_fragment(run_dir: Path, task: int, rows: list[dict]) -> None:
    frag_dir = run_dir / f"{run_dir.name}_task_{task}"
    frag_dir.mkdir(parents=True, exist_ok=True)
    with (frag_dir / "results.jsonl").open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


@pytest.fixture
def fake_registry(monkeypatch) -> list[str]:
    """Pin registry order to alpha < beta < gamma on the consolidate module."""
    order = ["alpha", "beta", "gamma"]
    consolidate_mod = sys.modules["mlsys.search.consolidate"]
    monkeypatch.setattr(consolidate_mod, "load_specs", lambda: dict.fromkeys(order))
    return order


def _fragments_for_pool(run_dir: Path, r2: dict[str, tuple[float, float]]) -> None:
    """One task per model, written in *reverse* registry order (task index still 0..n)."""
    for task, (model, (frozen_r2, finetune_r2)) in enumerate(reversed(list(r2.items()))):
        _write_fragment(
            run_dir,
            task,
            [_row(model, "frozen", frozen_r2), _row(model, "finetune", finetune_r2)],
        )


def test_merge_orders_blocks_by_registry_and_recomputes_regret(tmp_path, fake_registry) -> None:
    run_dir = tmp_path / "777"
    _fragments_for_pool(run_dir, {"alpha": (0.5, 0.6), "beta": (0.9, 0.4), "gamma": (0.7, 0.8)})
    result = consolidate_run(run_dir)

    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
    assert [(r["model"], r["strategy"]) for r in rows] == [
        ("alpha", "frozen"),
        ("beta", "frozen"),
        ("gamma", "frozen"),
        ("alpha", "finetune"),
        ("beta", "finetune"),
        ("gamma", "finetune"),
    ]
    # Rows pass through verbatim.
    assert rows[0] == _row("alpha", "frozen", 0.5)

    payload = json.loads((run_dir / "regret.json").read_text())
    assert payload["proxy_ranking"] == ["beta", "gamma", "alpha"]
    assert payload["metric"] == "r2"
    assert payload["regret_estimator"] == "point_estimate"
    # Best finetune r2 is gamma's 0.8; at B=1 the shortlist is {beta} → regret 0.8-0.4.
    curve = {p["budget"]: p["regret"] for p in payload["curve"]}
    assert curve[1] == pytest.approx(0.4)
    assert curve[2] == pytest.approx(0.0)  # gamma enters at B=2
    assert curve[3] == pytest.approx(0.0)
    assert result.regret_path == run_dir / "regret.json"
    assert result.dataset == "synthetic"


def test_tiebreak_follows_registry_order(tmp_path, fake_registry) -> None:
    # Equal frozen r2 everywhere, fragments arriving in reverse registry order:
    # the stable sort must break ties by registry (models.yaml) order, not task order.
    run_dir = tmp_path / "778"
    _fragments_for_pool(run_dir, {"alpha": (0.5, 0.1), "beta": (0.5, 0.2), "gamma": (0.5, 0.3)})
    consolidate_run(run_dir)
    payload = json.loads((run_dir / "regret.json").read_text())
    assert payload["proxy_ranking"] == ["alpha", "beta", "gamma"]


def test_csv_export_names_and_columns(tmp_path, fake_registry) -> None:
    run_dir = tmp_path / "779"
    _fragments_for_pool(run_dir, {"alpha": (0.5, 0.6), "beta": (0.9, 0.4), "gamma": (0.7, 0.8)})
    result = consolidate_run(run_dir, hidden=512)

    stem = "779_synthetic_fulleval_3_model_MLP_512"
    assert sorted(p.name for p in result.csv_paths) == sorted(
        f"{stem}_{kind}.csv" for kind in ("frozen", "finetune", "regret")
    )

    # The filename grammar must round-trip through the analysis loader's parser.
    from mlsys.analysis.loader import parse_filename

    parsed = parse_filename(result.csv_paths[0])
    assert (parsed.run_id, parsed.dataset, parsed.strategy) == ("779", "synthetic", "fulleval")
    assert (parsed.head, parsed.kind) == ("MLP_512", "frozen")

    frozen_csv = (run_dir / f"{stem}_frozen.csv").read_text().splitlines()
    header = frozen_csv[0].split(",")
    # Flat W&B-table layout: metrics/timing inlined, curves dropped, extras kept.
    assert set(header) == set(flatten_row(_row("alpha", "frozen", 0.5)).keys())
    assert "head_train_curve" not in header
    assert [line.split(",")[header.index("model")] for line in frozen_csv[1:]] == [
        "alpha",
        "beta",
        "gamma",
    ]

    regret_csv = (run_dir / f"{stem}_regret.csv").read_text().splitlines()
    assert regret_csv[0] == "budget,regret,normalized_regret"
    assert len(regret_csv) == 1 + 3


def test_idempotent_rerun_and_keep_last_dedupe(tmp_path, fake_registry) -> None:
    run_dir = tmp_path / "780"
    _fragments_for_pool(run_dir, {"alpha": (0.5, 0.6), "beta": (0.9, 0.4)})
    # A retried task appended a second (model, strategy) pair — keep-last must win.
    _write_fragment(run_dir, 0, [_row("gamma", "frozen", 0.1), _row("gamma", "finetune", 0.1)])
    _write_fragment(run_dir, 0, [_row("gamma", "frozen", 0.95), _row("gamma", "finetune", 0.99)])

    first = consolidate_run(run_dir)
    bytes_first = (run_dir / "results.jsonl").read_bytes()
    regret_first = (run_dir / "regret.json").read_bytes()

    payload = json.loads(regret_first)
    assert payload["frozen_r2"]["gamma"] == pytest.approx(0.95)  # keep-last
    assert payload["proxy_ranking"][0] == "gamma"

    second = consolidate_run(run_dir)
    assert (run_dir / "results.jsonl").read_bytes() == bytes_first
    assert (run_dir / "regret.json").read_bytes() == regret_first
    assert [r["model"] for r in second.rows] == [r["model"] for r in first.rows]


def test_no_fragments_falls_back_to_consolidated_file(tmp_path, fake_registry) -> None:
    run_dir = tmp_path / "781"
    _fragments_for_pool(run_dir, {"alpha": (0.5, 0.6), "beta": (0.9, 0.4)})
    consolidate_run(run_dir, cleanup=True)
    assert not list(run_dir.glob("*_task_*"))

    # Post-cleanup rerun: no fragments, but the merged results.jsonl is reused.
    result = consolidate_run(run_dir)
    assert [r["model"] for r in result.rows if r["strategy"] == "frozen"] == ["alpha", "beta"]


def test_failure_modes(tmp_path, fake_registry) -> None:
    with pytest.raises(FileNotFoundError):
        consolidate_run(tmp_path / "empty")

    # frozen without finetune → incomplete pool.
    run_dir = tmp_path / "782"
    _write_fragment(run_dir, 0, [_row("alpha", "frozen", 0.5), _row("alpha", "finetune", 0.6)])
    _write_fragment(run_dir, 1, [_row("beta", "frozen", 0.9)])
    with pytest.raises(ValueError, match="beta"):
        consolidate_run(run_dir)
    # Failure must leave the fragment dirs in place even with cleanup requested.
    with pytest.raises(ValueError):
        consolidate_run(run_dir, cleanup=True)
    assert len(list(run_dir.glob("*_task_*"))) == 2

    result = consolidate_run(run_dir, allow_partial=True)
    assert result.regret_path is None
    assert result.summary is None
    assert not (run_dir / "regret.json").exists()
    assert (run_dir / "results.jsonl").exists()
    assert len(result.csv_paths) == 2  # no regret CSV

    # Fragments disagreeing on dataset are a hard error.
    run_dir = tmp_path / "783"
    _write_fragment(run_dir, 0, [_row("alpha", "frozen", 0.5, dataset="one")])
    _write_fragment(run_dir, 1, [_row("beta", "frozen", 0.5, dataset="two")])
    with pytest.raises(ValueError, match="dataset"):
        consolidate_run(run_dir)


# --- End-to-end parity: single-node full_eval vs per-task runs + consolidation ---

torch = pytest.importorskip("torch")

from mlsys.datasets import LoadedDataset, Row  # noqa: E402
from mlsys.datasets.registry import DatasetSpec  # noqa: E402
from mlsys.head import HeadTrainConfig  # noqa: E402
from mlsys.models.registry import _ADAPTERS, ModelSpec, register_adapter  # noqa: E402
from mlsys.search.full_eval import run_full_eval  # noqa: E402


class _FakeBackbone:
    def __init__(self, spec: ModelSpec, device: str) -> None:
        self.name = spec.name
        self.embedding_dim = spec.embedding_dim
        self._device = device
        gen = torch.Generator().manual_seed(spec.embedding_dim)
        self._proj = torch.randn(64, spec.embedding_dim, generator=gen)

    def encode(self, texts: list[str]):
        rows = []
        for t in texts:
            h = abs(hash(t)) % 64
            v = torch.zeros(64)
            v[h] = 1.0
            rows.append(v)
        return (torch.stack(rows, dim=0) @ self._proj).to(self._device)


class _FakeSplit:
    def __init__(self, rows: list[Row]) -> None:
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)


class _FakeDataset:
    def __init__(self, name: str = "synthetic") -> None:
        n = 20
        self.spec = DatasetSpec(
            name=name,
            hf_repo="local/fake",
            splits={"train": "train", "val": "val", "test": "test"},
            target_column="y",
            target_type="regression",
            text_template="{text}",
        )
        self.splits = {
            "train": _FakeSplit([Row(text=f"t{i}", target=float(i)) for i in range(n)]),
            "val": _FakeSplit([Row(text=f"v{i}", target=float(i)) for i in range(n // 2)]),
            "test": _FakeSplit([Row(text=f"x{i}", target=float(i)) for i in range(n // 2)]),
        }

    def split(self, name: str):
        return self.splits[name]


def test_consolidated_regret_matches_single_node(tmp_path, monkeypatch) -> None:
    from typing import cast

    register_adapter("fake_loader", lambda spec, device: _FakeBackbone(spec, device))
    try:
        specs = {
            name: ModelSpec(
                name=name, hf_repo="local/fake", loader="fake_loader", embedding_dim=dim
            )
            for name, dim in (("fake-a", 8), ("fake-b", 16))
        }
        for mod_name in ("mlsys.search.full_eval", "mlsys.search.consolidate"):
            monkeypatch.setattr(sys.modules[mod_name], "load_specs", lambda: specs)
        # Head seeds normally come from unseeded torch RNG; pin them so the head
        # training (which reseeds the global RNG per repeat) is fully deterministic
        # and per-model scores agree between the two run shapes.
        for mod_name in ("mlsys.search.full_eval", "mlsys.search.runner"):
            monkeypatch.setattr(sys.modules[mod_name], "make_seeds", lambda n: list(range(n)))

        head_cfg = HeadTrainConfig(epochs=3, batch_size=4, lr=1e-2)
        dataset = cast(LoadedDataset, _FakeDataset())

        def run(out_dir: Path, model_names: list[str] | None = None) -> None:
            run_full_eval(
                dataset,
                out_dir,
                model_names=model_names,
                device="cpu",
                batch_size=4,
                head_config=head_cfg,
                head_repeats=2,
            )

        dir_a = tmp_path / "999"
        run(dir_a)

        dir_b = tmp_path / "999b"
        for task, name in enumerate(specs):
            run(dir_b / f"999b_task_{task}", model_names=[name])
        result = consolidate_run(dir_b)

        regret_a = (dir_a / "regret.json").read_text()
        regret_b = (dir_b / "regret.json").read_text()
        # Byte-identical modulo the dataset-independent payload — dataset name matches
        # too, so the whole file must be equal.
        assert regret_a == regret_b

        rows_a = [json.loads(x) for x in (dir_a / "results.jsonl").read_text().splitlines()]
        rows_b = [json.loads(x) for x in (dir_b / "results.jsonl").read_text().splitlines()]
        for row in (*rows_a, *rows_b):
            row.pop("timing")
        assert rows_a == rows_b
        assert result.run_name == "999b_synthetic_fulleval_2_model_FCH"
    finally:
        _ADAPTERS.pop("fake_loader", None)
