from ezlm import TigerConfig, TigerForCausalLM, TigerTokenizer
from ezlm import load_tokenized_dataset, print_model_size
from ezlm.variables import *

from datasets import load_dataset, concatenate_datasets
import fire

from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

from transformers import LlamaTokenizer, AutoTokenizer


base_model = 'quantumaikr/tiger-0.5b'
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer = TigerTokenizer.from_pretrained(base_model)
model = TigerForCausalLM.from_pretrained(base_model, cache_dir="hub", device_map="auto")

train_dataset = load_dataset("lcw99/wikipedia-korean-20221001", split="train[:2]", cache_dir="hub")
wiki_train_dataset = load_tokenized_dataset(tokenizer, 'kwiki', train_dataset)

train_dataset = load_dataset('junelee/sharegpt_deepl_ko', data_files='ko_alpaca_style_dataset.json', split="train[:]", cache_dir="hub")
sharegpt_train_dataset = load_tokenized_dataset(tokenizer, 'alpaca', train_dataset)

train_dataset = load_dataset('nlpai-lab/kullm-v2', split="train[:10000]", cache_dir="hub")
kullm_train_dataset = load_tokenized_dataset(tokenizer, 'alpaca', train_dataset)

train_dataset = load_dataset('FreedomIntelligence/alpaca-gpt4-korean', split="train[:]", cache_dir="hub")
alpaca_gpt4_train_dataset = load_tokenized_dataset(tokenizer, 'wizard', train_dataset)


all_dataset = concatenate_datasets([wiki_train_dataset, sharegpt_train_dataset, kullm_train_dataset, alpaca_gpt4_train_dataset])


# 훈련 시작
train_args = TrainingArguments(
    output_dir="result",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    save_steps=2000,
    logging_steps=100,
    save_strategy='steps',
    save_safetensors=True,
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=train_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    train_dataset=all_dataset,
)

trainer.train()
