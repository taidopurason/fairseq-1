import argparse
import logging
from typing import Iterable, List, OrderedDict, Tuple

import torch

from fairseq.file_io import PathManager

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("extend_fs_model")

ENCODER_EMBED_NAME = "encoder.embed_tokens.weight"
DECODER_EMBED_NAME = "decoder.embed_tokens.weight"
OUT_PROJ_NAME = "decoder.output_projection.weight"

EMBED_LAYER_NAMES = (
    ENCODER_EMBED_NAME,
    DECODER_EMBED_NAME,
    OUT_PROJ_NAME,  # might be present sometimes
)

ILLEGAL_CHARS = {" ", "\n", "\r", ""}


def read_dict(path: str) -> List[Tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return [tuple(line.rstrip().split(" ")) for line in f]


def read_tokens(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip() for line in f]


def preprocess_tokens(
    tokens: Iterable[str], data_dict_tokens: Iterable[str]
) -> List[str]:
    data_dict_tokens = set(data_dict_tokens)
    return [
        token
        for token in list(OrderedDict.fromkeys(tokens))
        if token not in data_dict_tokens and token not in ILLEGAL_CHARS
    ]


def add_tokens(
    tokens_to_add, state_dict, model_dict_entries, data_dict_entries, init_std=None
):
    if len(tokens_to_add) == 0:
        logger.info(f"No tokens to add.")
        return
    embedding_dim = state_dict[EMBED_LAYER_NAMES[0]].size(-1)

    insert_idx = len(data_dict_entries)
    entries_to_add = list(zip(tokens_to_add, [1] * len(tokens_to_add)))
    model_dict_entries[:] = [
        *model_dict_entries[:insert_idx],
        *entries_to_add,
        *model_dict_entries[insert_idx:],
    ]
    data_dict_entries[:] = [*data_dict_entries, *entries_to_add]

    model_insert_idx = insert_idx + 4

    for embed_layer in EMBED_LAYER_NAMES:
        if embed_layer in state_dict:
            embeds = state_dict[embed_layer]
            logger.info(f"{embed_layer} {len(embeds)} entries before adding tokens")

            new_weights = torch.normal(
                torch.zeros(len(entries_to_add), embedding_dim),
                std=embedding_dim ** -0.5 if init_std is None else init_std,
            )
            state_dict[embed_layer] = (
                torch.cat(
                    [embeds[:model_insert_idx], new_weights, embeds[model_insert_idx:]]
                )
                .to(embeds.device, dtype=embeds.dtype)
                .clone()
            )

            logger.info(
                f"{embed_layer} {len(state_dict[embed_layer])} entries after adding tokens"
            )


def remove_tokens(tokens_to_filter, state_dict, model_dict_entries, data_dict_entries):
    if len(tokens_to_filter) == 0:
        logger.info(f"No tokens to remove")
        return

    num_entries = state_dict[EMBED_LAYER_NAMES[0]].size(0)

    mask = torch.ones(num_entries).bool()
    if len(tokens_to_filter) > 0:
        dict_mask = [token not in tokens_to_filter for (token, _) in data_dict_entries]
        # First 4 entries are bos, pad, eos, unk
        mask[4 : len(dict_mask) + 4] = torch.tensor(dict_mask).bool()

    for embed_layer in EMBED_LAYER_NAMES:
        if embed_layer in state_dict:
            embeds = state_dict[embed_layer]
            logger.info(f"{embed_layer} {len(embeds)} entries before removing tokens")

            state_dict[embed_layer] = embeds[mask].clone()
            logger.info(
                f"{embed_layer} {len(state_dict[embed_layer])} entries after removing tokens"
            )

    model_dict_entries[:] = [
        entry for is_kept, entry in zip(dict_mask, model_dict_entries) if is_kept
    ] + model_dict_entries[len(dict_mask) :]

    data_dict_entries[:] = [
        entry for is_kept, entry in zip(dict_mask, data_dict_entries) if is_kept
    ]


def share_layers(state_dict, share_all_embeddings, share_decoder_input_output_embed):
    if share_all_embeddings:
        shared_embed = None
        for name in EMBED_LAYER_NAMES:
            if name not in state_dict:
                continue

            if shared_embed is None:
                shared_embed = state_dict[name]
            else:
                state_dict[name] = shared_embed[:]

    elif (
        share_decoder_input_output_embed
        and OUT_PROJ_NAME in state_dict
        and DECODER_EMBED_NAME in state_dict
    ):
        state_dict[OUT_PROJ_NAME] = state_dict[DECODER_EMBED_NAME][:]


def save_dict_entries(dict_entries, path):
    with open(path, "w", encoding="utf-8") as f:
        for token, score in dict_entries:
            f.write(f"{token} {score}\n")


def main(args):
    data_dict_entries = read_dict(args.data_dict_path)
    model_dict_entries = (
        read_dict(args.model_dict_path)
        if args.model_dict_path is not None
        else data_dict_entries
    )

    data_dict_tokens = tuple(token for (token, _) in data_dict_entries)
    model_dict_tokens = tuple(token for (token, _) in model_dict_entries)

    tokens_to_filter = (
        set(read_tokens(args.remove_tokens_path))
        if args.remove_tokens_path is not None
        else {}
    )

    if args.add_tokens_path is not None:
        tokens_to_add = preprocess_tokens(
            read_tokens(args.add_tokens_path), data_dict_tokens
        )
    else:
        tokens_to_add = []

    if model_dict_tokens[: len(data_dict_tokens)] != data_dict_tokens:
        raise ValueError("The beginning of model dict must match with data dict")

    if len(set(model_dict_tokens).intersection(tokens_to_add)) != 0:
        raise ValueError(
            "The model dict contains the additional tokens as special tokens (e.g. lang symbols)"
        )

    checkpoint = torch.load(args.model_path, map_location=torch.device("cpu"))
    model_state_dict = checkpoint["model"]
    model_args = (
        checkpoint["args"]
        if "args" in checkpoint and checkpoint["args"] is not None
        else checkpoint["cfg"]["model"]
    )

    remove_tokens(
        tokens_to_filter, model_state_dict, model_dict_entries, data_dict_entries
    )
    add_tokens(
        tokens_to_add,
        model_state_dict,
        model_dict_entries,
        data_dict_entries,
        init_std=args.init_std,
    )

    if len(tokens_to_add) > 0 or len(tokens_to_filter) > 0:
        share_layers(
            model_state_dict,
            getattr(model_args, "share_all_embeddings", False),
            getattr(model_args, "share_decoder_input_output_embed", False),
        )

    with PathManager.open(args.model_out_path, "wb") as f:
        torch.save(checkpoint, f)

    if args.model_dict_out_path is not None:
        save_dict_entries(model_dict_entries, args.model_dict_out_path)

    save_dict_entries(data_dict_entries, args.data_dict_out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extend FairSeq model.",
    )

    parser.add_argument(
        "--model-path", required=True, help="Path to the FairSeq model."
    )
    parser.add_argument(
        "--data-dict-path", required=True, help="Path to the data dict."
    )
    parser.add_argument("--model-dict-path", help="Path to the model dict.")

    parser.add_argument("--model-out-path", required=True, help="Model output path.")
    parser.add_argument(
        "--data-dict-out-path", required=True, help="Data dict output path."
    )
    parser.add_argument("--model-dict-out-path", help="Model dict output path.")

    parser.add_argument(
        "--add-tokens-path",
        default=None,
        help="File which contains the tokens that will be added (1 token per line).",
    )

    parser.add_argument(
        "--remove-tokens-path",
        default=None,
        help="File which contains the tokens that will be removed (1 token per line).",
    )

    parser.add_argument(
        "--init-std", type=float, default=None, help="Initial std for new model weights"
    )

    args = parser.parse_args()
    main(args)
