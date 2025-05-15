#!/usr/bin/env python3
import argparse
import torch
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler, DataCollatorWithPadding
from torch.optim import Adam
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--max_predict_samples", type=int, default=-1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--model_path", type=str, default=None)
    return parser.parse_args()

def preprocess(batch, tokenizer):
    tokenized = tokenizer(
        batch['sentence1'],
        batch['sentence2'],
        truncation=True,
        padding=False,
        max_length=512
    )
    tokenized['labels'] = batch['label']
    return tokenized

def get_dataloaders(encoded_datasets, tokenizer, batch_size):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(encoded_datasets["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    eval_loader = DataLoader(encoded_datasets["validation"], batch_size=batch_size, collate_fn=data_collator)
    test_loader = DataLoader(encoded_datasets["test"], batch_size=1, shuffle=False)
    return train_loader, eval_loader, test_loader

def evaluate(model, dataloader, device, args):
    model.eval()
    all_preds, all_labels = [], []
    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())
    val_acc = accuracy_score(all_labels, all_preds)
    print("Validation Accuracy:", val_acc)
    with open("res.txt", "a") as f:
        f.write(f"epoch_num: {args.num_train_epochs}, lr: {args.lr}, batch_size: {args.batch_size}, eval_acc: {val_acc:.4f}\n")
    return val_acc

def predict(model, dataloader, raw_dataset, device, output_file="predictions.txt"):
    model.eval()
    predictions = []
    for i, batch in enumerate(tqdm(dataloader, desc="Predicting")):
        if isinstance(batch, list):
            batch = batch[0]
        input_dict = {
            k: torch.stack(v).to(device)
            for k, v in batch.items()
            if k != "labels"
        }
        with torch.no_grad():
            outputs = model(**input_dict)
        pred = torch.argmax(outputs.logits, dim=-1)
        pred_label = pred[0].item()
        sent1 = raw_dataset[i]["sentence1"]
        sent2 = raw_dataset[i]["sentence2"]
        predictions.append(f"{sent1}###{sent2}###{pred_label}")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(predictions))
    print(f"✅ Saved predictions to {output_file}")

def train(model, train_loader, eval_loader, args, device, tokenizer):
    optimizer = Adam(model.parameters(), lr=args.lr)
    num_training_steps = args.num_train_epochs * len(train_loader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    model.train()
    for epoch in range(args.num_train_epochs):
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            wandb.log({"loss": loss.item()})
    save_path = f"trained_model_lr{args.lr}_bs{args.batch_size}_ep{args.num_train_epochs}"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    return evaluate(model, eval_loader, device, args)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project="anlp-ex1", config=vars(args))
    dataset = load_dataset("glue", "mrpc")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoded = dataset.map(lambda x: preprocess(x, tokenizer), batched=True, remove_columns=dataset["train"].column_names)
    if args.max_train_samples != -1:
        encoded["train"] = encoded["train"].select(range(args.max_train_samples))
    if args.max_eval_samples != -1:
        encoded["validation"] = encoded["validation"].select(range(args.max_eval_samples))
    if args.max_predict_samples != -1:
        encoded["test"] = encoded["test"].select(range(args.max_predict_samples))
    train_loader, eval_loader, test_loader = get_dataloaders(encoded, tokenizer, args.batch_size)
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.to(device)
    if args.do_train:
        train(model, train_loader, eval_loader, args, device, tokenizer)
    if args.do_predict:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=2)
        model.to(device)
        predict(model, test_loader, dataset["test"], device)

if __name__ == "__main__":
    main()
