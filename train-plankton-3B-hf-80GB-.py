from ezlm import PlanktonForCausalLM
from ezlm import load_tokenized_dataset, print_model_size
from ezlm.variables import *

from datasets import load_dataset, concatenate_datasets

from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

from transformers import AutoTokenizer


from ezlm import PlanktonConfig, PlanktonForCausalLM
from ezlm import print_model_size

import torch

base_model = 'KoreanLM-3B'
tokenizer = AutoTokenizer.from_pretrained('quantumaikr/plankton_tokenizer')
# model = PlanktonForCausalLM.from_pretrained(base_model, cache_dir="hub")
model = PlanktonForCausalLM.from_pretrained(
    base_model, 
    torch_dtype=torch.bfloat16,
    cache_dir="hub")

train_dataset = load_dataset("lcw99/wikipedia-korean-20221001", split="train[56%:]", cache_dir="hub")
wiki_train_dataset = load_tokenized_dataset(tokenizer, 'kwiki', train_dataset)

# train_dataset = load_dataset('junelee/sharegpt_deepl_ko', data_files='ko_alpaca_style_dataset.json', split="train[:]", cache_dir="hub")
# sharegpt_train_dataset = load_tokenized_dataset(tokenizer, 'alpaca', train_dataset)

# train_dataset = load_dataset('nlpai-lab/kullm-v2', split="train[:]", cache_dir="hub")
# kullm_train_dataset = load_tokenized_dataset(tokenizer, 'alpaca', train_dataset, max_length=1024)

# train_dataset = load_dataset('FreedomIntelligence/alpaca-gpt4-korean', split="train[:]", cache_dir="hub")
# alpaca_gpt4_train_dataset = load_tokenized_dataset(tokenizer, 'wizard', train_dataset)


# all_dataset = concatenate_datasets([wiki_train_dataset, sharegpt_train_dataset, kullm_train_dataset, alpaca_gpt4_train_dataset])
all_dataset = concatenate_datasets([wiki_train_dataset])



# 훈련 시작
train_args = TrainingArguments(
    output_dir="result",
    num_train_epochs=1,
    # per_device_train_batch_size=14,
    per_device_train_batch_size=18,
    gradient_accumulation_steps=2,
    bf16=True,
    save_steps=1000,
    logging_steps=100,
    save_strategy='steps',
    save_safetensors=True,
    save_total_limit=1,
    # deepspeed=ZERO_2_SETTINGS,
    deepspeed=ZERO_3_SETTINGS
)

trainer = Trainer(
    model=model,
    args=train_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    train_dataset=all_dataset,
)

trainer.train()
trainer.model.save_pretrained('plankton-3B-hf')
trainer.model.push_to_hub('plankton-3B-hf')
