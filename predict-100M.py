from ezlm import PlanktonForCausalLM
from ezlm import load_tokenized_dataset, print_model_size
from ezlm.variables import *

from datasets import load_dataset, concatenate_datasets

from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

from transformers import AutoTokenizer


base_model = 'plankton-1B'
tokenizer = AutoTokenizer.from_pretrained('quantumaikr/plankton_tokenizer')
model = PlanktonForCausalLM.from_pretrained(base_model, cache_dir="hub", device_map="auto")


# model.eval()

message = """
[INST] 대한민국 조선의 역사에 대해 설명해주세요. [/INST]
"""
inputs = tokenizer(message, return_tensors="pt").to("cuda")
print(inputs)

output = model.generate(inputs['input_ids'],
                        do_sample=True,
                        temperature=0.3,
                        top_p=0.95,
                        top_k=40,
                        max_new_tokens=256, repetition_penalty=1.5,)


print(tokenizer.decode(output[0], skip_special_tokens=True))
