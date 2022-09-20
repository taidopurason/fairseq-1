import argparse
from enum import Enum
from importlib.util import module_from_spec, spec_from_file_location
from inspect import isclass
from typing import Callable


class InputTypes(str, Enum):
    token_per_line_txt = "token_per_line_txt"
    sp_vocab = "sp_vocab"
    sp_model = "sp_model"
    fs_dict = "fs_dict"


def load_filter(
    path: str, name: str, parser: argparse.ArgumentParser
) -> Callable[[str], bool]:
    filter_spec = spec_from_file_location("custom_filter_module", path)
    filter_module = module_from_spec(filter_spec)
    filter_spec.loader.exec_module(filter_module)
    filter = getattr(filter_module, name)

    if not isclass(filter):
        return filter

    if hasattr(filter, "add_args"):
        filter.add_args(parser)

    args = parser.parse_args()
    return filter(args)


def main(args, filter_method: Callable[[str], bool]):
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
        "--output",
        type=str,
        required=True,
        help="The path where to write the output tokens",
    )
    parser.add_argument(
        "--filter-path",
        type=str,
        required=True,
        help="The python file containing the filter function or class.",
    )
    parser.add_argument(
        "--filter-method-name",
        default="filter",
        type=str,
        help="The name of filter function or class in the python file.",
    )

    args, _ = parser.parse_known_args()
    filter_method = load_filter(args.filter_path, args.filter_method_name, parser)

    main(args, filter_method)
