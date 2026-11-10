from argparse import Namespace

import numpy as np
import pytest
import torch

from espnet2.tasks.asr import ASRTask
from espnet2.train.dataset import ESPnetDataset


def test_add_arguments():
    ASRTask.get_parser()


def test_add_arguments_help():
    parser = ASRTask.get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


def test_main_help():
    with pytest.raises(SystemExit):
        ASRTask.main(cmd=["--help"])


def test_main_print_config():
    with pytest.raises(SystemExit):
        ASRTask.main(cmd=["--print_config"])


def test_main_with_no_args():
    with pytest.raises(SystemExit):
        ASRTask.main(cmd=[])


def test_print_config_and_load_it(tmp_path):
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        ASRTask.print_config(f)
    parser = ASRTask.get_parser()
    parser.parse_args(["--config", str(config_file)])


@pytest.mark.parametrize("inference", [True, False])
def test_optional_data_names(inference):
    retval = ASRTask.optional_data_names(True, inference)

    assert "prompt" in retval
    assert "healthy_speech" in retval
    assert "sir_pair_mask" in retval


def test_tokenized_asr_builds_encoder_for_token_embedding_dimension():
    """Build with SSL D=6, K=3 clusters and a distinct ASR dimension E=4."""
    args = ASRTask.get_parser().parse_args(
        [
            "--input_size",
            "6",
            "--model",
            "tokenized_asr",
            "--model_conf",
            "{ctc_weight: 1.0, kmeans_weight: 0.0, diversity_weight: 0.0, "
            "quantizer_conf: {input_dim: 6, num_clusters: 3}, "
            "asr_embedding_conf: {embed_dim: 4, positional_dropout_rate: 0.0}, "
            "report_cer: false, report_wer: false}",
            "--encoder",
            "transformer",
            "--encoder_conf",
            "{output_size: 4, attention_heads: 1, linear_units: 4, "
            "num_blocks: 1, input_layer: linear}",
            "--decoder",
            "null",
            "--normalize",
            "null",
            "--specaug",
            "null",
        ]
    )
    args.token_list = ["<blank>", "<unk>", "a", "i", "<sos/eos>"]
    model = ASRTask.build_model(args)
    assert model.quantizer.input_dim == 6
    assert model.asr_embedding.output_size() == 4
    loss, _, _ = model(
        torch.randn(2, 10, 6),
        torch.tensor([10, 8]),
        torch.tensor([[2, 3], [3, 2]]),
        torch.tensor([2, 2]),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.asr_embedding.embed.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("pair_mask", [None, [1, 0], [0, 0]])
def test_gumbel_kmeans_sir_build_and_paired_training_data(tmp_path, pair_mask):
    tokens = tmp_path / "tokens.txt"
    tokens.write_text("<blank>\n<unk>\na\ni\n<sos/eos>\n")
    data_paths = []
    for name, lengths in [("speech", [10, 8]), ("healthy_speech", [7, 12])]:
        scp = tmp_path / f"{name}.scp"
        lines = []
        for b, length in enumerate(lengths):
            path = tmp_path / f"{name}_{b}.npy"
            np.save(path, np.random.randn(length, 6).astype(np.float32))
            lines.append(f"utt_{b} {path}\n")
        scp.write_text("".join(lines))
        data_paths.extend(["--train_data_path_and_name_and_type", f"{scp},{name},npy"])
    text = tmp_path / "text_int"
    text.write_text("utt_0 2 3\nutt_1 3 2\n")
    data_paths.extend(["--train_data_path_and_name_and_type", f"{text},text,text_int"])
    if pair_mask is not None:
        mask_file = tmp_path / "sir_pair_mask"
        mask_file.write_text(f"utt_0 {pair_mask[0]}\nutt_1 {pair_mask[1]}\n")
        data_paths.extend(
            [
                "--train_data_path_and_name_and_type",
                f"{mask_file},sir_pair_mask,text_int",
            ]
        )
    args = ASRTask.get_parser().parse_args(
        [
            "--token_list",
            str(tokens),
            "--input_size",
            "6",
            "--model",
            "gumbel_kmeans",
            "--model_conf",
            "{ctc_weight: 1.0, kmeans_weight: 0.0, diversity_weight: 0.0, "
            "quantizer_conf: {input_dim: 6, num_clusters: 3}, "
            "report_cer: false, report_wer: false}",
            "--model_conf",
            "sir_weight=0.2",
            "--encoder",
            "transformer",
            "--encoder_conf",
            "{output_size: 4, attention_heads: 1, linear_units: 4, "
            "num_blocks: 1, input_layer: linear}",
            "--decoder",
            "null",
            "--normalize",
            "null",
            "--specaug",
            "null",
            *data_paths,
        ]
    )
    dataset = ESPnetDataset(args.train_data_path_and_name_and_type)
    ASRTask.check_task_requirements(dataset, allow_variable_data_keys=False, train=True)
    collate = ASRTask.build_collate_fn(args, train=True)
    keys, batch = collate([dataset["utt_0"], dataset["utt_1"]])
    assert keys == ["utt_0", "utt_1"]
    assert batch["speech"].shape == (2, 10, 6)
    assert batch["healthy_speech"].shape == (2, 12, 6)
    assert batch["healthy_speech_lengths"].tolist() == [7, 12]
    if pair_mask is not None:
        assert batch["sir_pair_mask"].shape == (2, 1)
        assert batch["sir_pair_mask"].flatten().tolist() == pair_mask
        assert "sir_pair_mask_lengths" not in batch

    model = ASRTask.build_model(args)
    assert model.sir_weight == 0.2
    loss, stats, _ = model(**batch)
    loss.backward()
    assert torch.isfinite(stats["loss_sir"]).all()
    assert stats["sir_num_pairs"].item() == (2 if pair_mask is None else sum(pair_mask))
    assert model.quantizer.codebook.grad.abs().sum() > 0


def get_dummy_namespace():
    pytest.importorskip("whisper")
    return Namespace(
        token_type="whisper_multilingual",
        token_list=["<blank>", "a", "b", "c", "<space>", "<unk>"],
        input_size=40,
        frontend="frontend",
        frontend_conf={"n_fft": 51, "win_length": 40, "hop_length": 16},
        specaug="specaug",
        specaug_conf={"apply_time_warp": True, "time_mask_width_range": 4},
        normalize=None,
        normalize_conf=None,
        encoder="transformer",
        encoder_conf={
            "output_size": 40,
            "linear_units": 4,
            "num_blocks": 2,
        },
        decoder="transformer",
        decoder_conf={
            "linear_units": 4,
            "num_blocks": 2,
        },
        init=None,
        ctc_conf={},
        model_conf={},
    )


def test_build_preprocess_fn_prompt():
    pytest.importorskip("whisper")
    args = get_dummy_namespace()

    new_args = {
        "use_preprocessor": True,
        "bpemodel": "whisper_multilingual",
        "non_linguistic_symbols": None,
        "cleaner": None,
        "g2p": None,
        "use_lang_prompt": True,
        "use_nlp_prompt": False,
    }
    args.__dict__.update(new_args)

    preprocess = ASRTask.build_preprocess_fn(args, True)
    data = {}
    data["text"] = "a a"
    data["prompt"] = "a"
    preprocess._text_process(data)


def test_build_preprocess_fn_nlp_prompt():
    pytest.importorskip("whisper")
    args = get_dummy_namespace()

    new_args = {
        "use_preprocessor": True,
        "bpemodel": "whisper_multilingual",
        "non_linguistic_symbols": None,
        "cleaner": None,
        "g2p": None,
        "use_lang_prompt": False,
        "use_nlp_prompt": True,
    }
    args.__dict__.update(new_args)

    preprocess = ASRTask.build_preprocess_fn(args, True)
    data = {}
    data["text"] = "a a"
    data["prompt"] = "a a"
    preprocess._text_process(data)
