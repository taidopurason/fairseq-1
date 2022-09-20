from glob import glob
from typing import List


def _read_sp_file_tokens(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [token for line in f for token in line.rstrip().split(" ")]


class TrainedFilter:
    def __init__(self, args):
        self.tokens = {
            token
            for raw_path in args.train_files
            for path in glob(raw_path)
            for token in _read_sp_file_tokens(path)
        }

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--train-files",
            nargs="+",
            help="paths to files containing lines with space delimited tokens",
        )

    def __call__(self, token: str) -> bool:
        return token in self.tokens
