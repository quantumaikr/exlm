import re

def load_tokenized_dataset(tokenizer, data_style, train_dataset, max_length=512):
    train_dataset_mapped = None
    if data_style == 'kwiki':
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding=True,
                            return_overflowing_tokens=True,
                            truncation=True, max_length=max_length, stride=10, return_offsets_mapping=True)

    if data_style == "alpaca":
        def tokenize_function(examples):
            converted_examples = []
            for idx in range(len(examples['instruction'])):
                if len(examples['instruction'][idx]) < 10 or len(examples['output'][idx]) < 10:
                    continue
                
                text = ''
                if(len(examples['input'][idx]) ==0):  
                    text += f"<s> [INST] {examples['instruction'][idx]} [/INST]\n"
                else:
                    text += f"<s> [INST] {examples['instruction'][idx]} {examples['input'][idx]} [/INST]\n"

                text += f"{examples['output'][idx]} \n </s>"
                
                converted_examples.append(text)
            
            return tokenizer(converted_examples, padding=False,
                     return_overflowing_tokens=True,
                     truncation=True, max_length=max_length, stride=10)

    if data_style == "wizard":
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
                     truncation=True, max_length=max_length, stride=10)

    if data_style == "account":
        def tokenize_function(examples):
            converted_examples = []
            for item in examples['text']:
                # 주어진 text에서 [질의]와 [회신] 사이의 내용을 추출합니다.
                query_match = re.search(r"\[질의\]\s*\n(.*?)(?=\[회신\])", item, re.DOTALL)
                answer_match = re.search(r"\[회신\]\s*\n(.*?)(?=\[관련 회계기준\])", item, re.DOTALL)

                text = ''
                if query_match and answer_match:
                    query = query_match.group(1).strip()
                    answer = answer_match.group(1).strip()

                    text += f"<s> [INST] {query} [/INST]\n"
                    text += f"{answer} \n </s>"
                    
                else:
                    print("Couldn't extract query and answer from the text.")
                
                converted_examples.append(text)

            return tokenizer(converted_examples, padding=False,
                     return_overflowing_tokens=True,
                     truncation=True, max_length=max_length, stride=10)

    columns = list(train_dataset.features.keys())
    train_dataset_mapped = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=columns
    )
    return train_dataset_mapped
    
