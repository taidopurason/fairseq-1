import argparse
import itertools
import logging

import torch

from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.file_io import PathManager
from fairseq.utils import csv_int_list, csv_str_list

logger = logging.getLogger("modularize")


def main(args):
    modular_encoder_layers = args.encoder_modular_layers
    modular_decoder_layers = args.decoder_modular_layers
    lang_pairs = args.lang_pairs

    langs = [lang for lp in lang_pairs for lang in lp.split("-")]

    logger.info(f"reading model from {args.model_path}")
    with PathManager.open(args.model_path, "rb") as f:
        checkpoint = torch.load(f, map_location=torch.device("cpu"))

    language_specific_modules = {lang: {} for lang in langs}
    new_model_state = {}
    for key, value in checkpoint["model"].items():
        module_type, *rest = key.split(".")

        if rest[0] == "layers":
            layer_num = int(rest[1])
        else:
            layer_num = None

        if module_type == "encoder":
            modular_layer_nums = modular_encoder_layers
            get_module_lang = lambda x: x.split("-")[0]
        elif module_type == "decoder":
            modular_layer_nums = modular_decoder_layers
            get_module_lang = lambda x: x.split("-")[1]
        else:
            raise Exception("Unknown module type.")

        # clones weights for modular layers
        # otherwise uses the same underlying tensor
        clone_layer = (
            layer_num is not None and layer_num in modular_layer_nums
        ) or args.clone_all

        for lang_pair in args.lang_pairs:
            new_key = f"models.{lang_pair}.{key}"
            lang = get_module_lang(lang_pair)

            logger.debug(
                f"{key} -> {new_key} ({'cloned' if clone_layer else 'shared'})"
            )

            if clone_layer:
                if key not in language_specific_modules[lang]:
                    language_specific_modules[lang][key] = value.detach().clone()
                new_model_state[new_key] = language_specific_modules[lang][key]
            else:
                new_model_state[new_key] = value

    checkpoint["model"] = new_model_state

    if "cfg" in checkpoint and checkpoint["cfg"] is not None:
        cfg = checkpoint["cfg"]
    else:
        cfg = convert_namespace_to_omegaconf(checkpoint["args"])

    cfg["task"].name = args.new_task_name
    cfg["task"]._name = args.new_task_name

    cfg["model"]._name = args.new_model_name
    cfg["model"].name = args.new_model_name

    for cfg_name in ["model", "task"]:
        cfg[cfg_name].arch = args.new_arch_name
        cfg[cfg_name].lang_pairs = lang_pairs

    checkpoint["cfg"] = cfg
    checkpoint["args"] = None

    logger.info(f"writing model to {args.output_path}")
    with PathManager.open(args.output_path, "wb") as f:
        torch.save(checkpoint, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Modularize FairSeq model.",
    )

    parser.add_argument(
        "--model-path", required=True, help="Path to the FairSeq model."
    )

    parser.add_argument(
        "--output-path", required=True, help="Path of the resulting model."
    )
    parser.add_argument(
        "--lang-pairs",
        type=csv_str_list,
        required=False,
        default=None,
        help="Language pairs to create the modules for.",
    )
    parser.add_argument(
        "--langs",
        type=csv_str_list,
        required=False,
        default=None,
        help="Languages to create the modules for (Creates module)",
    )
    parser.add_argument(
        "--decoder-modular-layers",
        type=csv_int_list,
        required=False,
        default=list(),
        help="List of layers to modularize (clone) for the decoder. For other layers the weights are shared.",
    )
    parser.add_argument(
        "--encoder-modular-layers",
        type=csv_int_list,
        required=False,
        default=list(),
        help="List of layers to modularize (clone) for the encoder. For other layers the weights are shared.",
    )
    parser.add_argument(
        "--clone-all",
        action="store_true",
        default=False,
        help="Clones all weights instead of sharing.",
    )
    parser.add_argument(
        "--new-task-name",
        type=str,
        required=False,
        default="multilingual_translation_sampled",
        help="task to write into the model config",
    )
    parser.add_argument(
        "--new-model-name",
        type=str,
        required=False,
        default="multilingual_transformer",
        help="model class to write into the model config",
    )
    parser.add_argument(
        "--new-arch-name",
        type=str,
        required=False,
        default="multilingual_transformer",
        help="model arch to write into the model config",
    )

    args = parser.parse_args()
    if args.langs is None and args.lang_pairs is None:
        raise Exception("Please specify either langs or lang-pairs")
    elif args.lang_pairs is None:
        args.lang_pairs = list(map("-".join, itertools.permutations(args.langs, 2)))
        logger.info(
            f"inferring language-pairs from langs: {', '.join(args.lang_pairs)}"
        )

    main(args)
