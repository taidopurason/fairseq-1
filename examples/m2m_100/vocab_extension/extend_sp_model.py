import argparse
from typing import Callable, Iterable

from sentencepiece.sentencepiece_model_pb2 import ModelProto
from tqdm import tqdm

ILLEGAL_CHARS = {" ", "\n", "\r", ""}


def filter_sp_model(model: ModelProto, filter_method: Callable[[str], bool]):
    score_cntr = 0.0
    pieces = list(model.pieces)
    del model.pieces[:]

    for piece in pieces:
        if piece.type != 1:
            model.pieces.append(piece)
        elif filter_method(piece.piece):
            piece.score = score_cntr * -1.0
            model.pieces.append(piece)
            score_cntr += 1


def add_pieces(model: ModelProto, new_pieces: Iterable[str]):
    score = min(p.score for p in model.pieces) - 1
    vocab = {p.piece for p in model.pieces}

    for piece in new_pieces:
        if piece not in vocab and piece not in ILLEGAL_CHARS:
            model.pieces.append(ModelProto.SentencePiece(piece=piece, score=score))
            vocab.add(piece)
            score -= 1


def read_model(path: str):
    model = ModelProto()
    with open(path, "rb") as f:
        model.ParseFromString(f.read())
    return model


def main(args):
    model = read_model(args.model_path)

    if args.remove_tokens_path is not None:
        with open(args.remove_tokens_path, "r", encoding="utf-8") as f:
            remove_tokens = {line.rstrip() for line in f}
            filter_sp_model(model, lambda token: token not in remove_tokens)

    if args.add_tokens_path is not None:
        with open(args.add_tokens_path, "r", encoding="utf-8") as f:
            tokens_to_add = [line.rstrip() for line in f]
            add_pieces(model, tokens_to_add)

    with open(args.output_prefix + ".model", "wb") as f:
        f.write(model.SerializeToString())

    with open(args.output_prefix + ".vocab", "w", encoding="utf-8") as f:
        for p in model.pieces:
            f.write(f"{p.piece}\t{int(p.score)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extend SentencePiece model.",
    )
    parser.add_argument("--model-path", required=True, help="Path to the sp model.")
    parser.add_argument("--output-prefix", required=True, help="Output prefix.")
    parser.add_argument(
        "--add-tokens-path",
        default=None,
        help="File which contains the tokens that will be added (1 token per line).",
    )
    parser.add_argument(
        "--remove-tokens-path",
        default=None,
        help="File which contains the tokens that will be added (1 token per line).",
    )

    args = parser.parse_args()
    main(args)
