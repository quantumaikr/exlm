from ezlm import TigerConfig, TigerForCausalLM, TigerTokenizer
# from ezlm import load_dataset
from ezlm.variables import *

from datasets import load_dataset

tokenizer = TigerTokenizer.from_pretrained('quantumaikr/tiger-0.5b')

# sentences = [
#     "대한민국 국민 대한민국 국민 대한민국 국민 대한민국 국민",
#     "대한민국 국민 대한민국 국민 대한민국 국민 대한민국 국민"
# ]
# inputs = tokenizer(
#     sentences, truncation=True, return_overflowing_tokens=True, max_length=3, stride=1
# )

# print('inputs', len(inputs['input_ids']))
# print('inputs', inputs)


train_dataset = load_dataset('FreedomIntelligence/alpaca-gpt4-korean', split="train[:2]")  
train_dataset = train_dataset.remove_columns(['id'])

def tokenize_function(examples):
    converted_examples = []
    for conversations in examples["conversations"]:
        text = ''
        for item in conversations:
            if item['from'] == "human":
                text += f"<s> [INST] {item['value']} [/INST]\n"
            else:
                text += f"{item['value']} \n </s>"

        converted_examples.append(text)

    return tokenizer(converted_examples, padding=False,
                return_overflowing_tokens=True,
                truncation=True, max_length=20, stride=2)
    # return tokenizer(converted_examples)

train_dataset_mapped = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['conversations']
)

print('train_dataset_mapped', train_dataset_mapped)
# print('train_dataset_mapped', len(train_dataset_mapped['input_ids']))
# print('train_dataset_mapped', len(train_dataset_mapped[0]['input_ids']))
