from transformers import PreTrainedTokenizerFast

import os
from dotenv import load_dotenv 
load_dotenv('.env.dev')
HF_TOKEN = os.environ.get("HF_TOKEN")


tokenizer = PreTrainedTokenizerFast.from_pretrained('custom_tokenizer')
tokenizer.push_to_hub('plankton_tokenizer', token=HF_TOKEN)

