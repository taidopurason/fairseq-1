import argparse
from enum import Enum
from importlib.util import module_from_spec, spec_from_file_location
from typing import Callable


class InputTypes(str, Enum):
    token_per_line_txt = "token_per_line_txt"
    sp_vocab = "sp_vocab"
    sp_model = "sp_model"
    fs_dict = "fs_dict"


def main(args):
    filter_spec = spec_from_file_location("custom_filter_module", args.filter_path)
    filter_module = module_from_spec(filter_spec)
    filter_spec.loader.exec_module(filter_module)
    filter_method: Callable[[str], bool] = getattr(
        filter_module, args.filter_method_name
    )

    input_type = args.input_type
    if input_type == InputTypes.token_per_line_txt:
        with open(args.input, "r", encoding="utf-8") as f:
            tokens = tuple(token.rstrip() for token in f)
    elif input_type == InputTypes.sp_vocab:
        with open(args.input, "r", encoding="utf-8") as f:
            tokens = tuple(line.rstrip().split("\t")[0] for line in f)
    elif input_type == InputTypes.fs_dict:
        with open(args.input, "r", encoding="utf-8") as f:
            tokens = tuple(line.rstrip().split(" ")[0] for line in f)
    elif input_type == InputTypes.sp_model:
        from sentencepiece.sentencepiece_model_pb2 import ModelProto

        model = ModelProto()
        with open(args.input, "rb") as f:
            model.ParseFromString(f.read())

            tokens = tuple(p.piece for p in model.pieces if p.type == 1)
    else:
        raise ValueError("Unknown input type")

    with open(args.output, "w", encoding="utf-8") as f:
        for token in filter(lambda x: not filter_method(x), tokens):
            f.write(token + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a list of stuff to filter out.",
    )
    parser.add_argument("--input", required=True, help="Path to the input")
    parser.add_argument(
        "--input-type",
        default=InputTypes.token_per_line_txt.value,
        type=InputTypes,
        choices=[t.value for t in InputTypes],
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to the sp model."
    )
    parser.add_argument(
        "--filter-path", type=str, required=True, help="Filter out non-latin tokens."
    )
    parser.add_argument(
        "--filter-method-name",
        default="filter",
        type=str,
        help="Filter out non-latin tokens.",
    )

    args = parser.parse_args()
    main(args)
