import argparse
import json
from pathlib import Path

from novex.cli import run_params, write_params
from novex.genome import parse_motifs, parse_starts


def test_run_params_serialises_parsed_args(tmp_path):
    args = argparse.Namespace(
        query=Path("q.gtf"),
        direction="up",
        motifs=parse_motifs("GT-AG,AT-AC"),
        starts=parse_starts("ATG,CTG"),
        min_cds_ratio=0.5,
    )
    params = run_params(args, ["novex", "-q", "q.gtf"])

    path = tmp_path / "params.json"
    write_params(path, params)
    loaded = json.loads(path.read_text())

    assert loaded["command"] == ["novex", "-q", "q.gtf"]
    assert loaded["args"]["query"] == str(Path("q.gtf").resolve())
    assert loaded["args"]["motifs"] == ["AT-AC", "GT-AG"]
    assert loaded["args"]["starts"] == ["ATG", "CTG"]
    assert loaded["args"]["min_cds_ratio"] == 0.5
    assert "novex_version" in loaded and "started" in loaded


def test_run_params_keeps_no_screening_as_null():
    args = argparse.Namespace(motifs=parse_motifs("any"), starts=parse_starts("any"))
    params = run_params(args, ["novex"])
    assert params["args"]["motifs"] is None
    assert params["args"]["starts"] is None
