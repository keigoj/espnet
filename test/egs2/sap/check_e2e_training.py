"""GPU smoke test of the current SAP model using real sampler-sized batches.

Run from egs2/sap/asr3 with the training Python. This initializes a fresh model
and simulates the epoch-30/31 boundary; it does not run thirty training epochs.
No experiment checkpoint is read or overwritten.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from espnet2.samplers.num_elements_batch_sampler import NumElementsBatchSampler
from espnet2.schedulers.warmup_lr import WarmupLR
from espnet2.tasks.asr import ASRTask
from espnet2.train.dataset import ESPnetDataset
from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.reporter import Reporter
from espnet2.train.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="conf/train_e2e_gumbel_kmeans_wavlm_large_4090_amp_1200k.yaml",
    )
    parser.add_argument("--output", required=True)
    cli = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(0)
    np.random.seed(0)
    assert torch.cuda.is_available(), "An accessible CUDA device is required"
    args = ASRTask.get_parser().parse_args(
        [
            "--config",
            cli.config,
            "--token_list",
            "data/en_token_list/bpe_unigram5000/tokens.txt",
            "--bpemodel",
            "data/en_token_list/bpe_unigram5000/bpe.model",
            "--ngpu",
            "1",
            "--staged_freeze_param",
            "frontend",
            "--staged_freeze_param",
            "quantizer",
            "--staged_freeze_until_epoch",
            "30",
            "--output_dir",
            str(Path(cli.output).parent),
        ]
    )
    assert args.model == "tokenized_asr" and args.use_amp
    torch.backends.cudnn.deterministic = args.cudnn_deterministic
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    sampler = NumElementsBatchSampler(
        args.batch_bins,
        [
            "exp/asr_stats_e2e_raw_en_bpe5000_sp/train/speech_shape",
            "exp/asr_stats_e2e_raw_en_bpe5000_sp/train/text_shape.bpe",
        ],
    )
    selected = [sampler.batch_list[len(sampler) // 2], sampler.batch_list[-1]]
    dataset = ESPnetDataset(
        [
            ("dump/e2e/raw/train_sp/wav.scp", "speech", "sound"),
            ("dump/e2e/raw/train_sp/text", "text", "text"),
        ],
        preprocess=ASRTask.build_preprocess_fn(args, train=True),
        keys_to_load=set().union(*map(set, selected)),
    )
    collate = ASRTask.build_collate_fn(args, train=True)
    batches = [collate([dataset[key] for key in keys]) for keys in selected]
    model = ASRTask.build_model(args).cuda()
    assert model.frontend.output_size() == 1024
    assert model.quantizer.num_clusters == 2000
    assert model.asr_embedding.output_size() == 512
    # Centroids are (K=2000, D=1024); verify initialization survived task build.
    initial = np.load(args.model_conf["quantizer_conf"]["initial_centroid_path"])
    torch.testing.assert_close(
        model.quantizer.codebook.detach().cpu(),
        torch.from_numpy(initial),
        rtol=0,
        atol=0,
    )
    for name, parameter in model.named_parameters():
        if any(name == p or name.startswith(p + ".") for p in args.freeze_param):
            parameter.requires_grad_(False)
    optimizer = ASRTask.build_optimizers(args, model)[0]
    # Simulate the LR scheduler position at the boundary as well. The weights
    # are freshly initialized, so this is an execution test, not a CER estimate.
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    scheduler = WarmupLR(
        optimizer,
        **args.scheduler_conf,
        last_epoch=30 * (len(sampler) // args.accum_grad) - 1,
    )
    scaler = torch.cuda.amp.GradScaler()
    options = Trainer.build_options(args)
    Trainer._validate_staged_freeze_options(model, options)
    parameters = dict(model.named_parameters())
    watched = {
        "embedding": "asr_embedding.embed.weight",
        "centroids": "quantizer.codebook",
        "wavlm": next(
            name
            for name in parameters
            if name.startswith("frontend.") and name.endswith("layers.0.fc1.weight")
        ),
        "encoder": next(
            name
            for name in parameters
            if name.startswith("encoder.") and parameters[name].ndim == 2
        ),
        "decoder": next(
            name
            for name in parameters
            if name.startswith("decoder.") and parameters[name].ndim == 2
        ),
        "cnn": next(
            name
            for name in parameters
            if name.startswith(args.freeze_param[0]) and name.endswith("weight")
        ),
    }
    report = {
        "config": cli.config,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "fresh_weights_simulated_epoch_boundary": True,
        "batch_bins": args.batch_bins,
        "accum_grad": args.accum_grad,
        "batches": [
            {"utterances": len(keys), "samples": batch["speech_lengths"].tolist()}
            for keys, batch in batches
        ],
        "watched_parameters": watched,
        "phases": [],
    }
    reporter = Reporter()
    for epoch in (30, 31):
        Trainer._set_staged_freeze_state(model, epoch, options)
        before = {
            key: parameters[name].detach().clone() for key, name in watched.items()
        }
        gradients = {}
        assignments = []

        def record_gradient(key, grad):
            assert grad.is_cuda and torch.isfinite(grad).all(), key
            gradients[key] = max(gradients.get(key, 0.0), grad.float().norm().item())

        handles = [
            parameters[name].register_hook(
                lambda grad, key=key: record_gradient(key, grad)
            )
            for key, name in watched.items()
            if parameters[name].requires_grad
        ]

        def record_assignment(module, inputs, output):
            # B utterances, T SSL frames, K clusters; K probabilities normalize
            # to one on valid frames and zero on padded frames.
            probs = output.probabilities  # (B, T, K).
            lengths = inputs[1]  # (B,).
            valid = (
                torch.arange(probs.size(1), device=probs.device)[None]
                < lengths[:, None]
            )
            torch.testing.assert_close(probs.sum(-1), valid.to(probs.dtype))  # (B, T).
            assert torch.all((probs == 0) | (probs == 1))
            assert probs.is_cuda
            assignments.append(
                {"shape": list(probs.shape), "temperature": output.temperature.item()}
            )

        handles.append(model.quantizer.register_forward_hook(record_assignment))
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        # One real optimizer update per phase, preserving configured accumulation.
        iterator = [batches[i % len(batches)] for i in range(args.accum_grad)]
        with reporter.observe("train", epoch=epoch) as subreporter:
            invalid = Trainer.train_one_epoch(
                model,
                iterator,
                [optimizer],
                [scheduler],
                scaler,
                subreporter,
                None,
                options,
                DistributedOption(ngpu=1),
            )
        torch.cuda.synchronize()
        for handle in handles:
            handle.remove()
        assert not invalid, "The optimizer update was skipped"
        changes = {
            key: (parameters[name].detach() - before[key]).abs().max().item()
            for key, name in watched.items()
        }
        for key in ("embedding", "encoder", "decoder"):
            assert gradients[key] > 0 and changes[key] > 0, key
        assert changes["cnn"] == 0 and "cnn" not in gradients
        if epoch == 30:
            assert model.tokenizer_frozen
            assert all(
                changes[key] == 0 and key not in gradients
                for key in ("wavlm", "centroids")
            )
            assert model.quantizer.num_updates.item() == 0
        else:
            assert not model.tokenizer_frozen
            for key in ("wavlm", "centroids"):
                assert gradients[key] > 0 and changes[key] > 0, key
            assert model.quantizer.num_updates.item() == args.accum_grad
        stats = reporter.stats[epoch]["train"]
        assert np.isfinite(stats["loss"]) and np.isfinite(stats["grad_norm"])
        optimizer_steps = {
            key: int(optimizer.state.get(parameters[name], {}).get("step", 0))
            for key, name in watched.items()
        }
        for key in ("embedding", "encoder", "decoder"):
            assert optimizer_steps[key] == epoch - 29
        for key in ("wavlm", "centroids"):
            assert optimizer_steps[key] == int(epoch == 31)
        assert optimizer_steps["cnn"] == 0
        for index, assignment in enumerate(assignments):
            expected_temperature = (
                max(0.1, 2.0 * 0.999995**index) if epoch == 31 else 2.0
            )
            assert abs(assignment["temperature"] - expected_temperature) < 1.0e-6
        phase = {
            "epoch": epoch,
            "seconds": time.perf_counter() - started,
            "max_memory_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "max_memory_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "gradient_norms_scaled": gradients,
            "parameter_max_change": changes,
            "optimizer_steps": optimizer_steps,
            "quantizer_num_updates": model.quantizer.num_updates.item(),
            "assignments": assignments,
            "stats": {key: stats[key] for key in ("loss", "grad_norm", "optim0_lr0")},
        }
        report["phases"].append(phase)
        print(json.dumps(phase, ensure_ascii=False), flush=True)
    # Validation must use nearest-neighbor assignments and leave the counter alone.
    model.eval()
    _, batch = batches[0]
    speech = batch["speech"].cuda()  # (B, S), S padded waveform samples.
    lengths = batch["speech_lengths"].cuda()  # (B,).
    counter = model.quantizer.num_updates.item()
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        first, first_lengths = model._extract_feats(speech, lengths)
        second, _ = model._extract_feats(speech, lengths)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first.is_cuda and first.size(-1) == 512  # (B, T, E=512).
    assert model.quantizer.num_updates.item() == counter
    report["evaluation"] = {
        "deterministic": True,
        "shape": list(first.shape),
        "lengths": first_lengths.tolist(),
        "counter_unchanged": True,
    }
    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"PASS: {output}", flush=True)


if __name__ == "__main__":
    main()
