import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

RECIPE = Path(__file__).resolve().parents[3] / "egs2/sap/asr3"


@pytest.mark.parametrize("embedding", [None, {"embed_dim": 512}])
def test_frozen_30_epoch_base_uses_a_separate_embedding_experiment(tmp_path, embedding):
    """Exercise launcher options with a capture-only ASR stage, without training."""
    shutil.copy(RECIPE / "run_e2e.sh", tmp_path / "run_e2e.sh")
    (tmp_path / "utils").mkdir()
    shutil.copy(RECIPE / "utils/parse_options.sh", tmp_path / "utils/parse_options.sh")
    (tmp_path / "path.sh").write_text(":\n")
    (tmp_path / "asr_e2e.sh").write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$SAP_E2E_CAPTURE"\n'
    )
    (tmp_path / "asr_e2e.sh").chmod(0o755)
    config = {"model": "gumbel_kmeans", "model_conf": {}}
    if embedding is not None:
        config["model"] = "tokenized_asr"
        config["model_conf"]["asr_embedding_conf"] = embedding
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    (tmp_path / "decode.yaml").write_text("{}\n")
    np.save(tmp_path / "centers.npy", np.zeros((3, 6), dtype=np.float32))
    capture = tmp_path / "captured_args"
    command = [
        "bash",
        str(tmp_path / "run_e2e.sh"),
        "--asr_config",
        "config.yaml",
        "--inference_config",
        "decode.yaml",
        "--initial_centroid_path",
        "centers.npy",
        "--nclusters",
        "3",
        "--e2e_mode",
        "full",
        "--warmup_epochs",
        "30",
        "--max_epoch",
        "30",
        "--stage",
        "11",
        "--stop_stage",
        "11",
    ]
    result = subprocess.run(
        command,
        env={**os.environ, "SAP_E2E_CAPTURE": str(capture)},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    arguments = capture.read_text().splitlines()
    experiment = arguments[arguments.index("--asr_exp") + 1]
    tag = "_tokembed512" if embedding is not None else ""
    assert experiment == (
        f"exp/asr_e2e_wavlm_large_l21_gumbel_km3{tag}_full_warmup30_ep30_en"
    )
    asr_args = arguments[arguments.index("--asr_args") + 1]
    assert "--staged_freeze_until_epoch 30" in asr_args
    assert "--max_epoch 30" in asr_args
    invalid_command = command.copy()
    invalid_command[invalid_command.index("--warmup_epochs") + 1] = "31"
    invalid = subprocess.run(invalid_command, text=True, capture_output=True)
    assert invalid.returncode == 2
    assert "must not exceed" in invalid.stdout
