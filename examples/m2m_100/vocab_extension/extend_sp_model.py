import argparse
from typing import Callable, Iterable

from sentencepiece.sentencepiece_model_pb2 import ModelProto

ILLEGAL_CHARS = {" ", "\n", "\r"}


def is_special_piece(piece: ModelProto.SentencePiece) -> bool:
    return piece.type != 1


def filter_sp_model(
    model: ModelProto, filter_method: Callable[[str], bool]
) -> ModelProto:
    new_model = ModelProto()

    score_cntr = 0.0
    for piece in model.pieces:
        if is_special_piece(piece):
            new_model.pieces.append(
                ModelProto.SentencePiece(
                    piece=piece.piece, score=score_cntr, type=piece.type
                )
            )
        elif filter_method(piece.piece):
            new_model.pieces.append(
                ModelProto.SentencePiece(
                    piece=piece.piece, score=score_cntr * -1.0, type=piece.type
                )
            )
            score_cntr += 1
    return new_model


def add_pieces(model: ModelProto, new_pieces: Iterable[str]):
    score = min(p.score for p in model.pieces) - 1
    vocab = {p.piece for p in model.pieces}

    for ns in new_pieces:
        if ns not in vocab and ns not in ILLEGAL_CHARS:
            model.pieces.append(ModelProto.SentencePiece(piece=ns, score=score))
            score -= 1


def main(args):
    model = ModelProto()
    with open(args.model_path, "rb") as f:
        model.ParseFromString(f.read())

    if args.remove_tokens_path is not None:
        with open(args.remove_tokens_path, "r", encoding="utf-8") as f:
            remove_tokens = {line.rstrip() for line in f}
            new_model = filter_sp_model(model, lambda token: token not in remove_tokens)
    else:
        new_model = model

    if args.add_tokens_path is not None:
        with open(args.add_tokens_path, "r", encoding="utf-8") as f:
            tokens_to_add = [line.rstrip() for line in f]
            add_pieces(new_model, tokens_to_add)

    with open(args.output_prefix + ".model", "wb") as f:
        f.write(new_model.SerializeToString())

    with open(args.output_prefix + ".vocab", "w", encoding="utf-8") as f:
        for p in new_model.pieces:
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
