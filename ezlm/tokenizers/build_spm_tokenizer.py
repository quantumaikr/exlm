# import sentencepiece as spm

# from datasets import load_dataset, concatenate_datasets

# code_datasets = load_dataset("huggingface-course/codeparrot-ds-valid", split="validation[:]")
# code_datasets = code_datasets.rename_column('content', 'text')
# code_datasets = code_datasets.remove_columns(['repo_name', 'path', 'copies', 'size', 'license'])

# print('code_datasets', code_datasets)

# en_datasets = load_dataset("EleutherAI/wikitext_document_level", 'wikitext-103-v1', split="train[:]")
# en_datasets = en_datasets.rename_column('page', 'text')

# print('code_datasets', en_datasets)

# ko_datasets = load_dataset("lcw99/wikipedia-korean-20221001", split="train[:]")
# ko_datasets = ko_datasets.remove_columns(['id', 'url', 'title'])

# print('code_datasets', ko_datasets)

# all_dataset = concatenate_datasets([code_datasets, ko_datasets, en_datasets])


# # import os
# # with open('./text_for_spm.txt', 'w', encoding='utf-8') as f:
# #     for data in all_dataset:
# #         f.write(data['text'] + '\n')


# print(all_dataset)

# from tokenizers import SentencePieceBPETokenizer

# tokenizer = SentencePieceBPETokenizer()
# tokenizer.train_from_iterator(
#     all_dataset['text'],
#     vocab_size=42_000,
#     min_frequency=5,
#     show_progress=True,
#     limit_alphabet=1000,
#     special_tokens=["<s>","</s>","<unk>"]
# )

# from transformers import PreTrainedTokenizerFast

# transformer_tokenizer = PreTrainedTokenizerFast(
#     tokenizer_object=tokenizer
# )

# transformer_tokenizer.save_pretrained('korean_tokenizer')

# from transformers import PreTrainedTokenizerFast
# tokenizer = PreTrainedTokenizerFast.from_pretrained('korean_tokenizer')


from transformers import AutoTokenizer

new_tokenizer = AutoTokenizer.from_pretrained('korean_tokenizer')
print(new_tokenizer.bos_token_id)
print(new_tokenizer.eos_token_id)
print(new_tokenizer.unk_token_id)
print(new_tokenizer.pad_token_id)

new_tokenizer.save_pretrained('korean_tokenizer')



print(new_tokenizer.encode("대한민국 국민"))