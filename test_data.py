from ezlm import TigerConfig, TigerForCausalLM, TigerTokenizer
from ezlm import load_tokenized_dataset
from ezlm.variables import *

from datasets import load_dataset
import fire

from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


def main(
        base_model: str = 'quantumaikr/tiger-0.5b' 
    ):

    tokenizer = TigerTokenizer.from_pretrained(base_model)
    # model = TigerForCausalLM.from_pretrained(base_model, cache_dir="hub", device_map="auto")
    
    
    # train_dataset = load_dataset('junelee/sharegpt_deepl_ko', data_files='ko_alpaca_style_dataset.json', split="train[:1000]", cache_dir="hub")
    # train_dataset = load_tokenized_dataset(tokenizer, 'alpaca', train_dataset)

    train_dataset = load_dataset('nlpai-lab/kullm-v2', split="train[:10000]", cache_dir="hub")
    train_dataset = load_tokenized_dataset(tokenizer, 'alpaca', train_dataset)

    print(train_dataset)
    
    
    # # 훈련 시작
    # train_args = TrainingArguments(
    #     output_dir="result",
    #     num_train_epochs=1,
    #     per_device_train_batch_size=4,
    #     gradient_accumulation_steps=4,
    #     save_steps=2000,
    #     logging_steps=100,
    #     save_strategy='steps',
    #     save_safetensors=True,
    #     save_total_limit=1,
    # )

    # trainer = Trainer(
    #     model=model,
    #     args=train_args,
    #     data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    #     train_dataset=train_dataset,
    # )

    # trainer.train()
    


if __name__ == "__main__":
    fire.Fire(main)