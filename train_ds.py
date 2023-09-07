from ezlm import TigerConfig, TigerForCausalLM, TigerTokenizer
from ezlm import load_tokenized_dataset
from ezlm.variables import *

from datasets import load_dataset
import fire

from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


def main(
        base_model: str = 'quantumaikr/tiger-0.5b',
        epochs: int = 3,
        batch_size: int = 4,
        ga_steps: int = 4,
        save_steps: int = 1000,
        logging_steps: int = 20,    
        save_strategy: str = 'steps'
    ):

    tokenizer = TigerTokenizer.from_pretrained(base_model)
    model = TigerForCausalLM.from_pretrained(base_model, cache_dir="hub")
    
    
    # train_dataset = load_dataset('junelee/sharegpt_deepl_ko', data_files='ko_alpaca_style_dataset.json', split="train[:2]", cache_dir="hub")
    # train_dataset = load_tokenized_dataset(tokenizer, 'alpaca', train_dataset)
    
    train_dataset = load_dataset('FreedomIntelligence/alpaca-gpt4-korean', split="train[:]", cache_dir="hub")
    train_dataset = load_tokenized_dataset(tokenizer, 'wizard', train_dataset)
    
    # train_dataset = load_dataset('nlpai-lab/kullm-v2', split="train[:]", cache_dir="hub")
    # train_dataset = load_tokenized_dataset(tokenizer, 'alpaca', train_dataset)

    # train_dataset = load_dataset("lcw99/wikipedia-korean-20221001", split="train[:]", cache_dir="hub")
    # train_dataset = load_tokenized_dataset(tokenizer, 'kwiki', train_dataset)

    
    # 훈련 시작
    train_args = TrainingArguments(
        output_dir="result",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=ga_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        save_safetensors=True,
        save_total_limit=1,
        deepspeed=ZERO_2_SETTINGS,
        # deepspeed=ZERO_3_SETTINGS
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.model.save_pretrained('saved_model')
    


if __name__ == "__main__":
    fire.Fire(main)