from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from carbontracker.tracker import CarbonTracker
import traceback
import logging

app = FastAPI()

carbon_report = {}

class TrainRequest(BaseModel):
    epochs: int = 2
    dataset_size: int = 100
    model_name: str = "distilbert-base-uncased"

class CarbonReportResponse(BaseModel):
    emissions_kg: float
    duration_sec: float
    epochs_trained: int

def train_model(epochs: int, dataset_size: int, model_name: str):
    global carbon_report
    dataset = load_dataset("imdb")
    dataset = dataset.shuffle(seed=42)
    small_train = dataset["train"].select(range(dataset_size))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tokenize_fn(example):
        return tokenizer(example["text"], padding="max_length", truncation=True)
    tokenized_dataset = small_train.map(tokenize_fn, batched=True)
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset
    )

    tracker = CarbonTracker(epochs=epochs)
    for epoch in range(epochs):
        tracker.epoch_start()
        trainer.train(resume_from_checkpoint=False)
        tracker.epoch_end()
    tracker.stop()

    carbon_report = {
        "emissions_kg": tracker.emissions,
        "duration_sec": tracker.duration * 60,
        "epochs_trained": epochs
    }

@app.post("/train", response_model=CarbonReportResponse)
async def train(request: TrainRequest):
    global carbon_report
    try:
        train_model(request.epochs, request.dataset_size, request.model_name)
    except Exception as e:
        logging.error(f"Training failed: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
    if not carbon_report:
        raise HTTPException(status_code=500, detail="Training did not produce a report.")
    return carbon_report