

## 환경 설치 관련

!git clone https://github.com/quantumaikr/ezlm

!pip install -r requirements.txt

!pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7 scipy deepspeed

!chmod +x install.sh | ./install.sh

!huggingface-cli login --token XXXX

https://huggingface.co/settings/tokens


# 실행

!huggingface-cli login --token XXXX

!deepspeed --num_gpus=4 train.py



## 데이터세트 로딩

```python
train_dataset = load_dataset('junelee/sharegpt_deepl_ko', data_files='ko_dataset_2.json', split="train[:]", cache_dir="hub")
train_dataset = load_tokenized_dataset(tokenizer, 'alpaca', train_dataset)

train_dataset = load_dataset('FreedomIntelligence/alpaca-gpt4-korean', split="train[:]", cache_dir="hub")
train_dataset = load_tokenized_dataset(tokenizer, 'alpaca', train_dataset)

train_dataset = load_dataset('nlpai-lab/kullm-v2', split="train[:]", cache_dir="hub")
train_dataset = load_tokenized_dataset(tokenizer, 'instruction', train_dataset)

train_dataset = load_dataset("lcw99/wikipedia-korean-20221001", split="train[:]", cache_dir="hub")
train_dataset = load_tokenized_dataset(tokenizer, 'kwiki', train_dataset)
```



## 훈련 코드

```python

from ezlm import TigerConfig, TigerForCausalLM, TigerTokenizer
from ezlm import load_tokenized_dataset
from ezlm.variables import *

from datasets import load_dataset

from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

base_model = 'quantumaikr/tiger-0.5b'
tokenizer = TigerTokenizer.from_pretrained(base_model)
model = TigerForCausalLM.from_pretrained(base_model, cache_dir="hub", device_map="auto")


# 데이터 로딩
train_dataset = load_dataset('FreedomIntelligence/alpaca-gpt4-korean', split="train[:]", cache_dir="hub")
train_dataset = load_tokenized_dataset(tokenizer, 'alpaca', train_dataset)


# 훈련 시작
train_args = TrainingArguments(
    output_dir="result",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
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
    train_dataset=train_dataset,
)

trainer.train()

```


# 모델 배포

model.save_pretrained('tiger-0.5b', safe_serialization=True, max_shard_size='3GB')

model.push_to_hub('tiger-0.5b', safe_serialization=True, max_shard_size='3GB', private=True)




# 모델 추론

```python
model.eval()

message = """
### <s> [INST] 미국 여행을 위해 준비해야 할 내용을 설명해주세요. [/INST]
### 
"""
inputs = tokenizer(message, return_tensors="pt").to("cuda")
output = model.generate(**inputs,
                        do_sample=True,
                        temperature=0.4,
                        top_p=0.95,
                        top_k=40,
                        max_new_tokens=256, repetition_penalty=1.5,
                        eos_token_id=tokenizer.eos_token_id)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```