from ezlm import PlanktonForCausalLM
from ezlm import load_tokenized_dataset, print_model_size
from ezlm.variables import *

from datasets import load_dataset, concatenate_datasets

from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

from transformers import AutoTokenizer

base_model = 'quantumaikr/plankton-100M'
tokenizer = AutoTokenizer.from_pretrained('quantumaikr/plankton_tokenizer')
model = PlanktonForCausalLM.from_pretrained(base_model, cache_dir="hub", device_map="auto")


train_dataset = load_dataset('csv',data_files='inquiry.csv', split="train[:]", cache_dir="hub")
account_train_dataset = load_tokenized_dataset(tokenizer, 'account', train_dataset)


# # all_dataset = concatenate_datasets([wiki_train_dataset, sharegpt_train_dataset, kullm_train_dataset, alpaca_gpt4_train_dataset])
# all_dataset = concatenate_datasets([alpaca_gpt4_train_dataset])



# # 훈련 시작
# train_args = TrainingArguments(
#     output_dir="result",
#     num_train_epochs=1,
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=1,
#     save_steps=3000,
#     logging_steps=100,
#     save_strategy='steps',
#     save_safetensors=True,
#     save_total_limit=1,
# )

# trainer = Trainer(
#     model=model,
#     args=train_args,
#     data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
#     train_dataset=all_dataset,
# )

# trainer.train()
# trainer.model.save_pretrained('plankton-100M')
# trainer.model.push_to_hub('plankton-100M')
