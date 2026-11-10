import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "egs2/sap/asr2/local/prepare_sir_mild.py"
SPEC = importlib.util.spec_from_file_location("prepare_sir_mild", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_doc(root, speaker, recordings):
    data = {
        "Contributor ID": speaker,
        "Files": [
            {
                "Ratings": [
                    {"Dimension Description": dimension, "Level": level}
                    for dimension, level in ratings
                ]
            }
            for ratings in recordings
        ],
    }
    (root / f"{speaker}.json").write_text(json.dumps(data))


def write_data(root, rows):
    root.mkdir()
    for filename, column in [("text", 2), ("utt2spk", 1), ("utt2uniq", 3)]:
        (root / filename).write_text(
            "".join(f"{row[0]} {row[column]}\n" for row in rows)
        )
    paths = []
    for utt, *_ in rows:
        wav = root / f"{utt}.wav"
        wav.touch()
        paths.append(f"{utt} {wav}\n")
    (root / "wav.scp").write_text("".join(paths))


def test_sap_intelligibility_spelling_missing_and_equal_recording_weights(tmp_path):
    write_doc(
        tmp_path,
        "speaker",
        [
            [(" Intelligbility ", "1"), ("Intelligibility", "3")],
            [("Intelligbility (overall)", "4"), ("Naturalness", "7")],
            [
                ("Intelligibility", ""),
                ("Intelligibility", "nan"),
                ("Intelligibility", "0"),
                ("Intelligibility", "8"),
            ],
        ],
    )
    score = MODULE.speaker_score(tmp_path / "speaker.json")
    assert score == {"mean": 3.0, "count": 2, "min": 2.0, "max": 4.0, "reason": "rated"}
    assert MODULE.speaker_score(tmp_path / "missing.json")["mean"] is None
    write_doc(tmp_path, "unrated", [[("Naturalness", "1")]])
    assert MODULE.speaker_score(tmp_path / "unrated.json")["mean"] is None


def test_prepare_mild_training_references_masks_and_no_validation_leakage(tmp_path):
    docs = tmp_path / "doc"
    docs.mkdir()
    write_doc(docs, "mild", [[("Intelligbility ", "1")], [("Intelligibility", "3")]])
    write_doc(docs, "affected", [[("Intelligibility", "3")]])
    write_doc(docs, "validation", [[("Intelligibility", "3")]])
    train = tmp_path / "train"
    valid = tmp_path / "valid"
    write_data(
        train,
        [
            ("ref", "mild", "SAME WORDS", "ref"),
            ("source", "affected", "same   words", "source"),
            ("sp0.9-source", "sp0.9-affected", "same words", "source"),
            ("no-match", "affected", "different words", "no-match"),
            ("unknown", "unrated", "same words", "unknown"),
        ],
    )
    write_data(
        valid,
        [
            ("dev", "validation", "same words", "dev"),
            ("dev-unmatched", "validation", "only in validation", "dev-unmatched"),
        ],
    )
    output = tmp_path / "sir"
    summary = MODULE.prepare(docs, train, valid, tmp_path, output, 2.0)
    assert (output / "mild_speakers.txt").read_text() == "mild\n"
    assert summary["train_speakers"] == 3
    assert summary["unrated_train_speakers"] == 1
    assert summary["splits"]["train"]["paired"] == 2
    assert summary["splits"]["valid"]["paired"] == 1
    masks = MODULE.read_keyed(output / "train/sir_pair_mask")
    assert masks == {
        "ref": "0",
        "source": "1",
        "sp0.9-source": "1",
        "no-match": "0",
        "unknown": "0",
    }
    reference_path = str((train / "ref.wav").resolve())
    refs = MODULE.read_keyed(output / "train/healthy.scp")
    assert refs["source"] == refs["sp0.9-source"] == reference_path
    assert MODULE.read_keyed(output / "valid/healthy.scp")["dev"] == reference_path
    assert (output / "speaker_scores.tsv").is_file()
    assert json.loads((output / "summary.json").read_text()) == summary
    second = tmp_path / "sir_second"
    MODULE.prepare(docs, train, valid, tmp_path, second, 2.0)
    assert (second / "train/pairs.tsv").read_text() == (
        output / "train/pairs.tsv"
    ).read_text()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        MODULE.prepare(docs, train, valid, tmp_path, output, 2.0)


@pytest.mark.parametrize("value", [0.0, 8.0, float("nan")])
def test_prepare_rejects_invalid_threshold(tmp_path, value):
    with pytest.raises(ValueError, match="finite value"):
        MODULE.prepare(tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, value)
