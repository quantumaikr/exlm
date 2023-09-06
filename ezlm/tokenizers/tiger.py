from transformers import AutoTokenizer

import os
from dotenv import load_dotenv 


load_dotenv('.env.dev')
HF_TOKEN = os.environ.get("HF_TOKEN")

class TigerTokenizer():
    
    def from_pretrained(model_path: str = 'quantumaikr/tiger', cache_dir="hub"):
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN, cache_dir=cache_dir)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
        
        return tokenizer
    