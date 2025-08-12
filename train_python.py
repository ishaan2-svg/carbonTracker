from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
from carbontracker.tracker import CarbonTracker

# Load dataset (1000 samples to keep it light)
print("Loading dataset...")
dataset = load_dataset("imdb")
dataset = dataset.shuffle(seed=42)
small_train = dataset["train"].select(range(1000))

# Tokenize
print("Tokenizing...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def tokenize_fn(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)
tokenized_dataset = small_train.map(tokenize_fn, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Load model
print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    logging_dir="./logs",
    logging_steps=10,  # Removed evaluation_strategy
    save_strategy="no"
)

# Set up trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# CarbonTracker wrapped training
print("Starting training with CarbonTracker...")
tracker = CarbonTracker(epochs=2)

for epoch in range(2):
    tracker.epoch_start()
    trainer.train(resume_from_checkpoint=False)
    tracker.epoch_end()

tracker.stop()
print("Training done! Carbon report saved.")
