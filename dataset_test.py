from ezlm import TigerConfig, TigerForCausalLM, TigerTokenizer


from datasets import load_dataset
import re

base_model: str = 'quantumaikr/tiger-0.5b' 


tokenizer = TigerTokenizer.from_pretrained(base_model)

ko_datasets = load_dataset("lcw99/wikipedia-korean-20221001", split="train[:1000]", cache_dir="hub")
ko_datasets = ko_datasets.remove_columns(['id', 'url', 'title'])

tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    text_list = []
    for text in examples["text"]:
        processed_text = re.split('참고 문헌|같이 보기|외부 링크', text)[0]
        text_list.append(processed_text)
    return tokenizer(text_list, padding=True,
                     return_overflowing_tokens=True,
                     truncation=True, max_length=100, stride=50)

tokenized_datasets = ko_datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)
tokenized_datasets