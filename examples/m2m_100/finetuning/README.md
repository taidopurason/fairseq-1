# Fine-tuning M2M-100

To fine-tune m2m-100 with your own dataset, preprocess the dataset with SentencePiece and then binarize it with
fairseq-preprocess as shown in the ```m2m_100``` directory's readme.

For example:

```
# Apply SentencePiece encoding
python /path/to/fairseq/scripts/spm_encode.py \
    --model spm.128k.model \
    --output_format=piece \
    --inputs=/path/to/input/file/here \
    --outputs=/path/to/output/file/here

# Binarize
fairseq-preprocess \
    --source-lang $src --target-lang $tgt \
    --testpref spm.$src-$tgt \
    --thresholdsrc 0 --thresholdtgt 0 \
    --destdir data_bin \
    --srcdict data_dict.128k.txt --tgtdict data_dict.128k.txt
```

An example fine-tuning script for 418M version of M2M-100:

```
fairseq-train ${bin_dir} \
  --task translation_multi_simple_epoch --arch transformer_wmt_en_de_big \
  --max-epoch 40 \
  --fixed-dictionary model_dict.128k.txt \
  --finetune-from-model 418M_last_checkpoint.pt \
  --save-dir checkpoints \
  --lang-pairs en-et,et-en,et-fi,fi-et,fi-en,en-fi \
  --max-tokens 3840 --update-freq 1 \
  --encoder-normalize-before --decoder-normalize-before \
  --share-decoder-input-output-embed --share-all-embeddings \
  --encoder-embed-dim 1024 --decoder-embed-dim 1024 \
  --encoder-ffn-embed-dim 4096 --decoder-ffn-embed-dim 4096 \
  --encoder-layers 12 --decoder-layers 12 \
  --encoder-attention-heads 16 --decoder-attention-heads 16 \
  --encoder-layerdrop 0.05 --decoder-layerdrop 0.05 \
  --lang-tok-style multilingual --decoder-langtok --encoder-langtok src \
  --attention-dropout 0.1 --activation-dropout 0.0 --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --lr '5e-05' \
  --lr-scheduler polynomial_decay --total-num-update 490000 --power 1.0 --end-learning-rate 0.0 --warmup-updates 0 \
  --criterion cross_entropy \
  --clip-norm 1.0 \
  --ddp-backend=no_c10d --num-workers 1
```

Where ```${bin_dir}``` is the resulting dataset from binarizing with ```fairseq-preprocess```. Feel free to change the
learning rate, the lr scheduler, max epochs, max tokens, etc.
