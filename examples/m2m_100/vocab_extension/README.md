**Generate a list of tokens to remove from the models:**

```
python find_tokens_to_remove.py \
--input spm.128k.model \
--input-type sp_model \
--output tokens_to_remove.txt \
--filter-path latin_filter.py \
```
`--filter-path` takes path to the Python file containing
a method named `filter` which takes token string as an input
and outputs a boolean value (True if token is kept False if removed)

The scripts outputs a file where each line contains a token that should be removed.

**Modify SentencePiece BPE model:**
```
python extend_sp_model.py \
--model-path spm.128k.model \
--output-prefix spm.filtered \
--add-tokens-path tokens_to_add.txt \
--remove-tokens-path tokens_to_remove.txt \
```


**Modify FairSeq model:**
```
python extend_fs_model.py \
--add-tokens-path tokens_to_add.txt \
--remove-tokens-path tokens_to_remove.txt \
--model-path 418M_last_checkpoint.pt \
--data-dict-path data_dict.128k.txt \
--model-dict-path model_dict.128k.txt \
--model-out-path 418M_last_checkpoint.mod.pt \
--data-dict-out-path data_dict.mod.txt \
--model-dict-out-path model_dict.mod.txt \
```

`--add-tokens-path` and `--remove-tokens-path` are optional arguments for these commands
and they can be used independently.
