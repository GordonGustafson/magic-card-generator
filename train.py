import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset

from dataset import card_dict_to_training_string


# Tokenization function
def tokenize_function(card_dict, tokenizer):
    text = card_dict_to_training_string(card_dict)
    result = tokenizer(text, padding="max_length", truncation=True, max_length=512)
    result["labels"] = result["input_ids"]
    return result


# Fine-tuning GPT-2
def fine_tune_gpt2(model_name="openai-community/gpt2"):
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name)
    # There's both "train" and "train_clean" splits. Haven't looked in to which to use yet.
    dataset = load_dataset("MechaCroc/magic-the-gathering")
    dataset = dataset["train"].select(range(100)).train_test_split(test_size=0.2, seed=42)

    print(dataset)

    # Tokenize dataset
    tokenized_datasets = dataset.map(lambda card_dict: tokenize_function(card_dict, tokenizer))
    print(tokenized_datasets)

    # Set format for PyTorch
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )

    # Train model
    trainer.train()

    # Save model
    trainer.save_model("./fine_tuned_gpt2")
    tokenizer.save_pretrained("./fine_tuned_gpt2")


if __name__ == "__main__":
    fine_tune_gpt2()
