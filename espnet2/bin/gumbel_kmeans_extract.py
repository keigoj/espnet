#!/usr/bin/env python3

import argparse
import logging
import sys
from itertools import groupby
from typing import Optional, Sequence, Tuple, Union

import torch
from typeguard import typechecked

from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.tasks.asr import ASRTask
from espnet2.tasks.ssl import SSLTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.utils.cli_utils import get_commandline_args


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract hard units from a trained Gumbel K-means model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task",
        choices=("ssl", "asr"),
        required=True,
        help="Task used to train the checkpoint",
    )
    parser.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    parser.add_argument("--train_config", type=str, required=True)
    parser.add_argument(
        "--model_file",
        type=str_or_none,
        required=True,
        help="Checkpoint path, or 'none' to use the model's initial weights",
    )
    parser.add_argument("--key_file", type=str_or_none)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--allow_variable_data_keys", type=str2bool, default=False)
    parser.add_argument("--remove_repeated", type=str2bool, default=False)
    parser.add_argument(
        "--loss_index",
        type=int,
        default=0,
        help="Gumbel K-means loss index for an SSL checkpoint",
    )
    parser.add_argument("--ngpu", type=int, default=0, choices=(0, 1))
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=("float16", "float32", "float64", "bfloat16"),
    )
    parser.add_argument(
        "--log_level",
        type=lambda value: value.upper(),
        default="INFO",
        choices=("ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
    )
    return parser


@typechecked
def extract_units(
    task: str,
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    train_config: str,
    model_file: Optional[str],
    key_file: Optional[str],
    output_dir: str,
    allow_variable_data_keys: bool,
    remove_repeated: bool,
    loss_index: int,
    ngpu: int,
    num_workers: int,
    batch_size: int,
    dtype: str,
    log_level: Union[int, str],
) -> None:
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    task_class = SSLTask if task == "ssl" else ASRTask
    device = "cuda" if ngpu == 1 else "cpu"
    model, train_args = task_class.build_model_from_file(
        train_config, model_file, device
    )
    model.to(device=device, dtype=getattr(torch, dtype)).eval()
    if not hasattr(model, "encode_units"):
        raise TypeError(
            f"{type(model).__name__} does not provide encode_units(); "
            "use a Gumbel K-means SSL loss or ASR model"
        )

    iterator = task_class.build_streaming_iterator(
        data_path_and_name_and_type=data_path_and_name_and_type,
        key_file=key_file,
        batch_size=batch_size,
        dtype=dtype,
        num_workers=num_workers,
        preprocess_fn=task_class.build_preprocess_fn(train_args, False),
        collate_fn=task_class.build_collate_fn(train_args, False),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    with DatadirWriter(output_dir) as writer:
        unit_writer = writer["pseudo_labels.txt"]
        for keys, batch in iterator:
            batch = to_device(batch, device=device)
            kwargs = {}
            if task == "ssl":
                kwargs["loss_index"] = loss_index
            units, unit_lengths = model.encode_units(
                speech=batch["speech"],
                speech_lengths=batch["speech_lengths"],
                **kwargs,
            )
            units = units.cpu()
            unit_lengths = unit_lengths.cpu()
            for key, sequence, length in zip(keys, units, unit_lengths):
                values = sequence[: int(length)].tolist()
                if remove_repeated:
                    values = [value for value, _ in groupby(values)]
                unit_writer[key] = " ".join(str(value) for value in values)


def main(cmd=None) -> None:
    print(get_commandline_args(), file=sys.stderr)
    args = get_parser().parse_args(cmd)
    extract_units(**vars(args))


if __name__ == "__main__":
    main()
