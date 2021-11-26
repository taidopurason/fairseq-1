from fairseq.models import register_model, register_model_architecture
from fairseq.models.multilingual_transformer import (
    MultilingualTransformerModel,
    base_multilingual_architecture,
)
from fairseq.models.transformer_align import TransformerAlignModel, transformer_align


@register_model("multilingual_transformer_align")
class MultilingualTransformerAlignModel(MultilingualTransformerModel):
    @staticmethod
    def build_submodel(encoder, decoder, args):
        return TransformerAlignModel(encoder, decoder, args)

    @staticmethod
    def get_encoder_decoder_model_class():
        return TransformerAlignModel

    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.models[self.keys[0]].forward_decoder(prev_output_tokens, **kwargs)


@register_model_architecture(
    "multilingual_transformer_align", "multilingual_transformer_align"
)
def multilingual_transformer_aligned(args):
    base_multilingual_architecture(args)
    transformer_align(args)
