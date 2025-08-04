from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments, Trainer, BitsAndBytesConfig
from datasets import load_dataset

data = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

tokenized_datasets = data.map(tokenize_function, batched=True)


model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
train_config = TrainingArguments(
    output_dir="opt/",
    learning_rate=1e-3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,
    deepspeed="deepseed/ds_config_stage3.json",
)
trainer = Trainer(
    model=model,
    args=train_config,
    train_dataset=tokenized_datasets["train"],  # Placeholder for training dataset
    eval_dataset=tokenized_datasets["test"],   # Placeholder for evaluation dataset
    tokenizer=tokenizer,
)

trainer.train()
