import argparse
import gc
from collections import OrderedDict

import torch

from fairseq.dataclass.utils import eval_dict


def load_state_dict(path, keep_prefix=None, rename_prefix=None):
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    state_dict = checkpoint["model"]
    if keep_prefix is None:
        return state_dict

    state_dict = OrderedDict(
        [(k, v) for k, v in state_dict.items() if k.startswith(keep_prefix)]
    )
    if rename_prefix is None:
        return state_dict

    return OrderedDict(
        [(rename_prefix + k[len(keep_prefix) :], v) for k, v in state_dict.items()]
    )


def rename_prefix(name, prefix, rename):
    if not name.startswith(prefix):
        return name

    if len(name) == len(prefix):
        return rename

    return rename + name[len(prefix) :]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-model-path")
    parser.add_argument("--decoder-model-path")
    parser.add_argument("--out-path")
    parser.add_argument("--encoder-prefix", required=False, default="encoder")
    parser.add_argument("--decoder-prefix", required=False, default="decoder")
    parser.add_argument("--encoder-rename-prefix", required=False, default=None)
    parser.add_argument("--decoder-rename-prefix", required=False, default=None)
    parser.add_argument(
        "--extra-rename-prefixes", required=False, default=None, type=eval_dict
    )

    args = parser.parse_args()

    encoder_state_dict = load_state_dict(
        args.encoder_model_path, args.encoder_prefix, args.encoder_rename_prefix
    )
    gc.collect()
    decoder_state_dict = load_state_dict(
        args.decoder_model_path, args.decoder_prefix, args.decoder_rename_prefix
    )
    gc.collect()
    final_state_dict = OrderedDict(
        list(encoder_state_dict.items()) + list(decoder_state_dict.items())
    )

    # option for additional renames
    if args.extra_rename_prefixes is not None:
        for prefix, rename in args.extra_rename_prefixes.items():
            final_state_dict = OrderedDict(
                (rename_prefix(k, prefix, rename), v)
                for k, v in final_state_dict.items()
            )

    state = torch.load(args.encoder_model_path, map_location=torch.device("cpu"))

    final_checkpoint = {
        "model": final_state_dict,
        # adding rest of the state for compatibility
        **{
            k: v for k, v in state.items() if k not in ("model", "last_optimizer_state")
        },
    }

    torch.save(final_checkpoint, args.out_path)
