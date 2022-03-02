# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from argparse import Namespace
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional

from fairseq import utils
from fairseq.models import (
    FairseqMultiModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    Embedding,
    TransformerDecoder,
    TransformerEncoder,
    TransformerModel,
    base_architecture,
)
from fairseq.modules import TransformerEncoderLayer
from fairseq.utils import check_lang_groups, list_of_csv_str_lists, safe_hasattr


class EncoderLayerSharingManager:
    def __init__(self, args: Namespace):
        self.n_shared_layers = args.shared_encoder_layers
        self.n_shared_lang_group_layers = args.shared_encoder_lang_group_layers
        self.lang_groups = args.lang_groups

        self.shared_layers = None
        self.shared_group_layers = {}

        if self.n_shared_lang_group_layers > 0 and self.lang_groups is not None:
            check_lang_groups(self.lang_groups)
            self.lang2group = dict(
                (lang, i) for i, langs in enumerate(self.lang_groups) for lang in langs
            )
        else:
            self.lang2group = None

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--shared-encoder-layers",
            type=int,
            help="number of top layers to share in the encoder",
        )
        parser.add_argument(
            "--shared-encoder-lang-group-layers",
            type=int,
            help="number of language group layers to share",
        )
        parser.add_argument(
            "--lang-groups",
            default=None,
            type=list_of_csv_str_lists,
            help="semicolon separated list of language groups. "
            "Each group consists of comma separated languages. "
            "Groups must be non-overlapping e.g. et,fi;lt,lv,ru;en,de",
        )

    @staticmethod
    def base_architecture(args):
        args.shared_encoder_layers = getattr(args, "shared_encoder_layers", 0)
        args.shared_encoder_lang_group_layers = getattr(
            args, "shared_encoder_lang_group_layers", 0
        )
        args.lang_groups = getattr(args, "lang_groups", None)

    @property
    def is_sharing(self):
        return self.lang2group is not None or self.n_shared_layers > 0

    def _get_shared_layers(self, lang: str) -> Dict[int, TransformerEncoderLayer]:
        # returns the shared layers for a language in the for of dict whose keys are layer indixes (e.g. 1, 2, -1, etc)
        shared_layers = {}
        if self.n_shared_layers > 0 and self.shared_layers is not None:
            shared_layers.update(
                {
                    -i - 1: module
                    for i, module in enumerate(reversed(self.shared_layers))
                }
            )
        if self.lang2group is not None and lang in self.lang2group:
            group = self.lang2group[lang]
            if group in self.shared_group_layers:
                shared_layers.update(
                    {
                        -self.n_shared_layers - i - 1: module
                        for i, module in enumerate(
                            reversed(self.shared_group_layers[group])
                        )
                    }
                )
        return shared_layers

    def _update_shared_layers(self, lang: str, model: TransformerEncoder):
        # Updates the internal state with shared layers of the specified lang
        if self.n_shared_layers > 0 and self.shared_layers is None:
            self.shared_layers = model.layers[-self.n_shared_layers :]

        if self.lang2group is not None and lang in self.lang2group:
            group = self.lang2group[lang]
            if group not in self.shared_group_layers:
                idx_from = -self.n_shared_lang_group_layers - self.n_shared_layers
                idx_to = -self.n_shared_layers if self.n_shared_layers > 0 else None
                self.shared_group_layers[group] = model.layers[idx_from:idx_to]

    def _write_shared_layers(
        self,
        model: TransformerEncoder,
        shared_layers: Dict[int, TransformerEncoderLayer],
    ):
        for i, module in shared_layers.items():
            model.layers[i] = module

    def share_layers(self, lang: str, model: TransformerEncoder):
        shared_layers = self._get_shared_layers(lang)
        if len(shared_layers) > 0:
            self._write_shared_layers(model, shared_layers)
        self._update_shared_layers(lang, model)


@register_model("multilingual_transformer")
class MultilingualTransformerModel(FairseqMultiModel):
    """Train Transformer models for multiple language pairs simultaneously.

    Requires `--task multilingual_translation`.

    We inherit all arguments from TransformerModel and assume that all language
    pairs use a single Transformer architecture. In addition, we provide several
    options that are specific to the multilingual setting.

    Args:
        --share-encoder-embeddings: share encoder embeddings across all source languages
        --share-decoder-embeddings: share decoder embeddings across all target languages
        --share-language-specific-embeddings: share encoder and decoder embeddings language-specifically
        --share-encoders: share all encoder params (incl. embeddings) across all source languages
        --share-decoders: share all decoder params (incl. embeddings) across all target languages
    """

    def __init__(self, encoders, decoders, args):
        super().__init__(encoders, decoders)
        self.args = args

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)
        EncoderLayerSharingManager.add_args(parser)
        parser.add_argument(
            "--share-encoder-embeddings",
            action="store_true",
            help="share encoder embeddings across languages",
        )
        parser.add_argument(
            "--share-decoder-embeddings",
            action="store_true",
            help="share decoder embeddings across languages",
        )
        parser.add_argument(
            "--share-encoders",
            action="store_true",
            help="share encoders across languages",
        )
        parser.add_argument(
            "--share-decoders",
            action="store_true",
            help="share decoders across languages",
        )
        parser.add_argument(
            "--share-language-specific-embeddings",
            action="store_true",
            help="share encoder and decoder embeddings between the encoder and the decoder of the same language",
        )
        parser.add_argument(
            "--reduced-state-dict",
            action="store_true",
            help="Save only the encoders and decoders, not every language pair (avoids duplicating the encoders/decoders).",
        )
        parser.add_argument(
            "--lang-group-modules",
            default=None,
            type=list_of_csv_str_lists,
            help="share language-group modules instead of language modules."
            "e.g. et,fi;de,en",
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        from fairseq.tasks.multilingual_translation import MultilingualTranslationTask

        assert isinstance(task, MultilingualTranslationTask)

        # make sure all arguments are present in older models
        base_multilingual_architecture(args)

        if not safe_hasattr(args, "max_source_positions"):
            args.max_source_positions = 1024
        if not safe_hasattr(args, "max_target_positions"):
            args.max_target_positions = 1024

        src_langs = [lang_pair.split("-")[0] for lang_pair in task.model_lang_pairs]
        tgt_langs = [lang_pair.split("-")[1] for lang_pair in task.model_lang_pairs]

        if args.share_encoders:
            args.share_encoder_embeddings = True
        if args.share_decoders:
            args.share_decoder_embeddings = True

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        # build shared embeddings (if applicable)
        shared_encoder_embed_tokens, shared_decoder_embed_tokens = None, None
        language_specific_embeddings = None

        # checks if the encoder and decoder embed dims and paths are equal (for shared embeddings)
        def check_encoder_decoder_embed_args_equal():
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )

        if args.share_all_embeddings:
            check_encoder_decoder_embed_args_equal()
            shared_encoder_embed_tokens = FairseqMultiModel.build_shared_embeddings(
                dicts=task.dicts,
                langs=task.langs,
                embed_dim=args.encoder_embed_dim,
                build_embedding=build_embedding,
                pretrained_embed_path=args.encoder_embed_path,
            )
            shared_decoder_embed_tokens = shared_encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            if args.share_encoder_embeddings:
                shared_encoder_embed_tokens = FairseqMultiModel.build_shared_embeddings(
                    dicts=task.dicts,
                    langs=src_langs,
                    embed_dim=args.encoder_embed_dim,
                    build_embedding=build_embedding,
                    pretrained_embed_path=args.encoder_embed_path,
                )
            if args.share_decoder_embeddings:
                shared_decoder_embed_tokens = FairseqMultiModel.build_shared_embeddings(
                    dicts=task.dicts,
                    langs=tgt_langs,
                    embed_dim=args.decoder_embed_dim,
                    build_embedding=build_embedding,
                    pretrained_embed_path=args.decoder_embed_path,
                )

            if args.share_language_specific_embeddings:
                if args.share_encoder_embeddings or args.share_decoder_embeddings:
                    raise ValueError(
                        "--share-language-specific-embeddings is not compatible with "
                        "--share-encoder-embeddings or --share-decoder-embeddings."
                    )
                check_encoder_decoder_embed_args_equal()
                language_specific_embeddings = {
                    lang: build_embedding(
                        task.dicts[lang],
                        args.encoder_embed_dim,
                        args.encoder_embed_path,
                    )
                    for lang in task.langs
                }

        # encoders/decoders for each language
        lang_encoders, lang_decoders = {}, {}

        encoder_sharing_manager = EncoderLayerSharingManager(args)

        def get_encoder(lang):
            if lang not in lang_encoders:
                if shared_encoder_embed_tokens is not None:
                    encoder_embed_tokens = shared_encoder_embed_tokens
                elif language_specific_embeddings is not None:
                    encoder_embed_tokens = language_specific_embeddings[lang]
                else:
                    encoder_embed_tokens = build_embedding(
                        task.dicts[lang],
                        args.encoder_embed_dim,
                        args.encoder_embed_path,
                    )
                encoder = cls._get_module_class(
                    True, args, task.dicts[lang], encoder_embed_tokens, src_langs
                )
                if encoder_sharing_manager.is_sharing:
                    encoder_sharing_manager.share_layers(lang, encoder)
                lang_encoders[lang] = encoder
            return lang_encoders[lang]

        def get_decoder(lang):
            if lang not in lang_decoders:
                if shared_decoder_embed_tokens is not None:
                    decoder_embed_tokens = shared_decoder_embed_tokens
                elif language_specific_embeddings is not None:
                    decoder_embed_tokens = language_specific_embeddings[lang]
                else:
                    decoder_embed_tokens = build_embedding(
                        task.dicts[lang],
                        args.decoder_embed_dim,
                        args.decoder_embed_path,
                    )
                lang_decoders[lang] = cls._get_module_class(
                    False, args, task.dicts[lang], decoder_embed_tokens, tgt_langs
                )
            return lang_decoders[lang]

        # shared encoders/decoders (if applicable)
        shared_encoder, shared_decoder = None, None
        if args.share_encoders:
            shared_encoder = get_encoder(src_langs[0])
        if args.share_decoders:
            shared_decoder = get_decoder(tgt_langs[0])

        if args.lang_group_modules:
            check_lang_groups(args.lang_group_modules)
            for lang_group in args.lang_group_modules:
                for lang in lang_group:
                    lang_encoders[lang] = get_encoder(lang_group[0])
                    lang_decoders[lang] = get_decoder(lang_group[0])

        encoders, decoders = OrderedDict(), OrderedDict()
        for lang_pair, src, tgt in zip(task.model_lang_pairs, src_langs, tgt_langs):
            encoders[lang_pair] = (
                shared_encoder if shared_encoder is not None else get_encoder(src)
            )
            decoders[lang_pair] = (
                shared_decoder if shared_decoder is not None else get_decoder(tgt)
            )

        return MultilingualTransformerModel(encoders, decoders, args)

    @classmethod
    def _get_module_class(cls, is_encoder, args, lang_dict, embed_tokens, langs):
        module_class = TransformerEncoder if is_encoder else TransformerDecoder
        return module_class(args, lang_dict, embed_tokens)

    def load_state_dict(self, state_dict, strict=True, model_cfg=None):
        if self._is_reduced_state_dict(state_dict):
            state_dict_subset = self.restore_reduced_state_dict(
                state_dict, self.models.keys()
            )
        else:
            state_dict_subset = state_dict.copy()
            for k, _ in state_dict.items():
                assert k.startswith("models.")
                lang_pair = k.split(".")[1]
                if lang_pair not in self.models:
                    del state_dict_subset[k]
        super().load_state_dict(state_dict_subset, strict=strict, model_cfg=model_cfg)

    @staticmethod
    def _is_reduced_state_dict(state_dict):
        return any(
            map(
                lambda k: k.startswith("encoders.") or k.startswith("decoders."),
                state_dict.keys(),
            )
        )

    @staticmethod
    def restore_reduced_state_dict(state_dict, lang_pairs):
        encoders = defaultdict(lambda: [])
        decoders = defaultdict(lambda: [])

        for k in state_dict.keys():
            module, lang, *rest = k.split(".")
            if module == "encoders":
                encoders[lang].append(".".join(rest))
            elif module == "decoders":
                decoders[lang].append(".".join(rest))
            else:
                raise ValueError(
                    f"state must belong to an encoder or a decoder. state={k}"
                )

        new_state_dict = OrderedDict()
        for lang_pair in lang_pairs:
            src, tgt = lang_pair.split("-")
            for k in encoders[src]:
                new_state_dict[f"models.{lang_pair}.encoder.{k}"] = state_dict[
                    f"encoders.{src}.{k}"
                ]
            for k in decoders[tgt]:
                new_state_dict[f"models.{lang_pair}.decoder.{k}"] = state_dict[
                    f"decoders.{tgt}.{k}"
                ]
        return new_state_dict

    @staticmethod
    def reduce_state_dict(state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            lang_pair, module, *rest = k.split(".")[1:]

            if module != "encoder" and module != "decoder":
                raise ValueError(
                    f"reduced state dict only works with encoder-decoder models. key={k}"
                )

            src, tgt = lang_pair.split("-")

            new_key = (
                f"{module}s.{src if module == 'encoder' else tgt}.{'.'.join(rest)}"
            )

            if new_key not in new_state_dict:
                new_state_dict[new_key] = v

        return new_state_dict

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        if self.args.reduced_state_dict:
            return self.reduce_state_dict(state_dict)
        return state_dict


@register_model_architecture("multilingual_transformer", "multilingual_transformer")
def base_multilingual_architecture(args):
    base_architecture(args)
    args.share_encoder_embeddings = getattr(args, "share_encoder_embeddings", False)
    args.share_decoder_embeddings = getattr(args, "share_decoder_embeddings", False)
    args.share_language_specific_embeddings = getattr(
        args, "share_language_specific_embeddings", False
    )
    args.share_encoders = getattr(args, "share_encoders", False)
    args.share_decoders = getattr(args, "share_decoders", False)
    args.reduced_state_dict = getattr(args, "reduced_state_dict", False)
    args.lang_group_modules = getattr(args, "lang_group_modules", None)
    EncoderLayerSharingManager.base_architecture(args)


@register_model_architecture(
    "multilingual_transformer", "multilingual_transformer_iwslt_de_en"
)
def multilingual_transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_multilingual_architecture(args)
