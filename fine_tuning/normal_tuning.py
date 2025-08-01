from peft import LoraConfig, TaskType, get_peft_model , prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, TrainingArguments , Trainer , AutoTokenizer , BitsAndBytesConfig
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", 
                                             load_in_4bit=True,
                                             device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
from peft import get_peft_model
from datasets import load_dataset
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

raw_datasets = load_dataset("imdb")

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

peft_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_threshold=6.0,
)
model = prepare_model_for_kbit_training(model, peft_config)


training_args = TrainingArguments(
    output_dir="model/",
    learning_rate=1e-3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
