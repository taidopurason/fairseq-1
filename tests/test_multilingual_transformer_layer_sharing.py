import argparse
import unittest
from typing import Any, List
from unittest.mock import Mock

import torch

from fairseq.models import multilingual_transformer
from fairseq.tasks.multilingual_translation import MultilingualTranslationTask
from tests.utils import dummy_dictionary


def mk_transformer(langs: List[str], lang_pairs: List[str], **extra_args: Any):
    overrides = {
        "encoder_embed_dim": 12,
        "encoder_ffn_embed_dim": 14,
        "decoder_embed_dim": 12,
        "decoder_ffn_embed_dim": 14,
        "encoder_attention_heads": 4,
        "decoder_attention_heads": 4,
        "dropout": 0,
        "attention_dropout": 0,
        "activation_dropout": 0,
        "encoder_layerdrop": 0,
        "lang_pairs": lang_pairs,
    }
    overrides.update(extra_args)
    args = argparse.Namespace(**overrides)

    dicts = {lang: dummy_dictionary(50) for lang in langs}

    task = Mock(
        spec=MultilingualTranslationTask,
        dicts=dicts,
        lang_pairs=lang_pairs,
        model_lang_pairs=lang_pairs,
        langs=langs,
    )

    multilingual_transformer.base_architecture(args)

    torch.manual_seed(0)

    return multilingual_transformer.MultilingualTransformerModel.build_model(args, task)


class MultilingualTransformerTestCase(unittest.TestCase):
    langs = ["xx", "yy", "zz"]
    lang_pairs = ["xx-yy", "yy-xx", "xx-zz", "zz-xx", "yy-zz", "zz-yy"]

    def test_sharing_language_specific_encoders_decoders(self):
        model = mk_transformer(self.langs, self.lang_pairs)

        for src in self.langs:
            encoder, decoder = None, None
            for tgt in filter(lambda x: x != src, self.langs):
                enc = model.models[f"{src}-{tgt}"].encoder
                if encoder is None:
                    encoder = enc
                else:
                    assert encoder == enc

                dec = model.models[f"{tgt}-{src}"].decoder
                if decoder is None:
                    decoder = dec
                else:
                    assert decoder == dec

                assert enc != model.models[f"{tgt}-{src}"].encoder
                assert dec != model.models[f"{src}-{tgt}"].decoder

    def test_sharing_language_specific_embeddings(self):
        model = mk_transformer(
            self.langs, self.lang_pairs, **{"share_language_specific_embeddings": True}
        )
        for src in self.langs:
            for tgt in filter(lambda x: x != src, self.langs):
                assert (
                    model.models[f"{src}-{tgt}"].encoder.embed_tokens
                    == model.models[f"{tgt}-{src}"].decoder.embed_tokens
                )
                assert (
                    model.models[f"{src}-{tgt}"].encoder.embed_tokens
                    != model.models[f"{tgt}-{src}"].encoder.embed_tokens
                )

    def test_sharing_encoder_top_layers(self):
        n_shared_top_layers = 2
        n_encoder_layers = 6
        model = mk_transformer(
            self.langs,
            self.lang_pairs,
            **{
                "shared_encoder_layers": n_shared_top_layers,
                "encoder_layers": n_encoder_layers,
            },
        )
        for n in range(n_encoder_layers):
            layer = None
            lang = None
            for lang_pair in self.lang_pairs:
                l = model.models[lang_pair].encoder.layers[n]
                lng = lang_pair.split("-")[0]
                if layer is None:
                    layer = l
                    lang = lng
                elif n < n_encoder_layers - n_shared_top_layers and lng != lang:
                    assert layer != l
                else:
                    assert layer == l

    def test_sharing_encoder_lang_groups(self):
        n_shared_language_group_layers = 2
        n_encoder_layers = 6

        lang_groups = [["xx", "yy"], ["zz"]]
        lang2group = {"xx": 0, "yy": 0, "zz": 1}

        model = mk_transformer(
            self.langs,
            self.lang_pairs,
            **{
                "shared_encoder_lang_group_layers": n_shared_language_group_layers,
                "lang_groups": lang_groups,
                "encoder_layers": n_encoder_layers,
            },
        )

        for n in range(n_shared_language_group_layers):
            modules = {}
            for lang_pair in self.lang_pairs:
                l = model.models[lang_pair].encoder.layers[-n - 1]
                lng = lang_pair.split("-")[0]
                group = lang2group[lng]
                if group not in modules:
                    modules[group] = l
                else:
                    assert modules[group] == l

    def test_sharing_encoder_top_layers_and_lang_groups(self):
        n_shared_language_group_layers = 2
        n_shared_top_layers = 2
        n_encoder_layers = 6

        lang_groups = [["xx", "yy"], ["zz"]]
        lang2group = {"xx": 0, "yy": 0, "zz": 1}

        model = mk_transformer(
            self.langs,
            self.lang_pairs,
            **{
                "shared_encoder_lang_group_layers": n_shared_language_group_layers,
                "lang_groups": lang_groups,
                "encoder_layers": n_encoder_layers,
                "shared_encoder_layers": n_shared_top_layers,
            },
        )

        for n in range(n_shared_top_layers):
            layer = None
            for lang_pair in self.lang_pairs:
                l = model.models[lang_pair].encoder.layers[-n - 1]
                if layer is None:
                    layer = l
                else:
                    assert layer == l

        for n in range(n_shared_language_group_layers):
            modules = {}
            for lang_pair in self.lang_pairs:
                l = model.models[lang_pair].encoder.layers[-n - 1 - n_shared_top_layers]
                lng = lang_pair.split("-")[0]
                group = lang2group[lng]
                if group not in modules:
                    modules[group] = l
                else:
                    assert modules[group] == l
