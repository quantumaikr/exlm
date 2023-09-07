
from ezlm import TigerConfig, TigerForCausalLM, TigerTokenizer
from ezlm import load_tokenized_dataset

from ezlm import print_model_size

from ezlm.variables import *

from datasets import load_dataset
import fire

from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments



base_model = 'quantumaikr/tiger-0.5b'
tokenizer = TigerTokenizer.from_pretrained(base_model)
model = TigerForCausalLM.from_pretrained(base_model, cache_dir="hub")
    


print_model_size(model)