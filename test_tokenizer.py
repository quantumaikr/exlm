from transformers import PreTrainedTokenizerFast, AutoTokenizer

tokenizer = PreTrainedTokenizerFast.from_pretrained('custom_tokenizer')


tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids('<s>')
tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids('</s>')
tokenizer.unk_token_id = tokenizer.convert_tokens_to_ids('<unk>')
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<pad>')

tokenizer.save_pretrained('custom_tokenizer')


# new_tokenizer = AutoTokenizer.from_pretrained('custom_tokenizer')
# print(new_tokenizer.bos_token_id)

# # new_tokenizer.save_pretrained('korean_tokenizer')

print(tokenizer.pad_token_id)

# print(tokenizer.encode("대한민국 국민"))