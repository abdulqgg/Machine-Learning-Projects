from datasets import load_dataset

eli5 = load_dataset('eli5_category', split='train[:5000]', trust_remote_code=True)

eli5 = eli5.train_test_split(test_size=0.2)

print(eli5['train'][0]['answers']['text'])

from transformers import AutoTokenizer

# print(eli5['train'][0]["answer.text"])

tokenizer = AutoTokenizer.from_pretrained('distilbert/distilgpt2')

def preprocess_function(examples):
    return tokenizer([" "].join(x) for x in examples["answers"]["text"])

tokenized_eli5 = eli5.map(
    preprocess_function ,
    batched=True,
    num_proc=4,
    remove_columns=eli5['train'].column_names,
)