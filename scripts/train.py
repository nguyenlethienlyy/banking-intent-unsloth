# =============================================================================
# train.py
# Fine-tunes a LLaMA-3.2-1B model with Unsloth + LoRA for banking intent
# classification on the BANKING77 subset.
#
# Usage:
#   python scripts/train.py --config configs/train.yaml
# =============================================================================

import argparse
import json
import os

import pandas as pd
import torch
import yaml
from datasets import Dataset
from sklearn.metrics import accuracy_score
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


# =============================================================================
# Prompt template  (EOS token appended at runtime after tokenizer is loaded)
# =============================================================================

PROMPT_TEMPLATE = (
    "### Instruction:\n"
    "Classify the banking query below into one of the intent categories.\n\n"
    "### Input:\n"
    "{text}\n\n"
    "### Response:\n"
    "{label}"
)


def build_prompt(text: str, label: str = "", eos: str = "") -> str:
    return PROMPT_TEMPLATE.format(text=text, label=label) + eos


# =============================================================================
# Dataset helpers
# =============================================================================

def load_and_tokenize(
    train_path: str,
    label_map: dict,
    tokenizer,
    max_seq_length: int,
) -> Dataset:
    df = pd.read_csv(train_path)
    id_to_label = {v: k for k, v in label_map.items()}

    eos = tokenizer.eos_token or ""

    texts = [
        build_prompt(row["text"], id_to_label[row["label"]], eos)
        for _, row in df.iterrows()
    ]

    # Tokenize with right-padding
    tokenizer.padding_side = "right"

    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=False,          # DataCollatorForSeq2Seq handles padding
        max_length=max_seq_length,
        return_tensors=None,
    )

    # labels = input_ids (causal LM)
    tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
    return Dataset.from_dict(tokenized)


# =============================================================================
# Evaluation on test set
# =============================================================================

def evaluate(model, tokenizer, label_map: dict, cfg: dict):
    test_path = cfg["train_data_path"].replace("train.csv", "test.csv")
    if not os.path.exists(test_path):
        print("  Test file not found, skipping evaluation.")
        return

    print("\n[Evaluation] Running on test set ...")
    df = pd.read_csv(test_path)

    FastLanguageModel.for_inference(model)

    y_true, y_pred = [], []

    for _, row in df.iterrows():
        prompt = build_prompt(row["text"], label="", eos="")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens  = 20,
                pad_token_id    = tokenizer.eos_token_id,
                do_sample       = False,
            )

        decoded  = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = decoded.split("### Response:")[-1].strip().lower()

        predicted_id = -1
        for label_name, label_id in label_map.items():
            if label_name.lower() in response:
                predicted_id = label_id
                break

        y_true.append(row["label"])
        y_pred.append(predicted_id)

    acc = accuracy_score(y_true, y_pred)
    print(f"  Test Accuracy: {acc * 100:.2f}%")
    return acc


# =============================================================================
# Main
# =============================================================================

def main(config_path: str):
    # ---------------------------------------------------------------- config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    with open(cfg["label_map_path"], "r") as f:
        label_map = json.load(f)

    print(f"[Config] Model      : {cfg['model_name']}")
    print(f"[Config] Num labels : {len(label_map)}")
    print(f"[Config] Epochs     : {cfg['num_epochs']}")

    # ----------------------------------------------------------------- model
    print("\n[1/4] Loading model and tokenizer ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = cfg["model_name"],
        max_seq_length = cfg["max_seq_length"],
        load_in_4bit   = cfg["load_in_4bit"],
        dtype          = None,
    )

    # ------------------------------------------------------------------ LoRA
    print("[2/4] Applying LoRA adapters ...")
    model = FastLanguageModel.get_peft_model(
        model,
        r                          = cfg["lora_r"],
        lora_alpha                 = cfg["lora_alpha"],
        lora_dropout               = cfg["lora_dropout"],
        target_modules             = cfg["target_modules"],
        bias                       = "none",
        use_gradient_checkpointing = "unsloth",
        random_state               = cfg["seed"],
    )

    # ------------------------------------------------------------------ data
    print("[3/4] Preparing and tokenizing dataset ...")
    train_dataset = load_and_tokenize(
        cfg["train_data_path"], label_map, tokenizer, cfg["max_seq_length"]
    )
    print(f"      Train samples : {len(train_dataset)}")

    # ------------------------------------------------------------ collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer    = tokenizer,
        model        = model,
        padding      = True,
        pad_to_multiple_of = 8,
    )

    # --------------------------------------------------------------- trainer
    print("[4/4] Starting training ...")

    training_args = TrainingArguments(
        output_dir                  = cfg["output_dir"],
        num_train_epochs            = cfg["num_epochs"],
        per_device_train_batch_size = cfg["per_device_train_batch_size"],
        gradient_accumulation_steps = cfg["gradient_accumulation_steps"],
        learning_rate               = cfg["learning_rate"],
        lr_scheduler_type           = cfg["lr_scheduler_type"],
        warmup_steps                = cfg["warmup_steps"],
        optim                       = cfg["optimizer"],
        weight_decay                = cfg["weight_decay"],
        fp16                        = cfg["fp16"],
        bf16                        = cfg["bf16"],
        logging_steps               = cfg["logging_steps"],
        save_strategy               = cfg["save_strategy"],
        seed                        = cfg["seed"],
        report_to                   = "none",
    )

    trainer = SFTTrainer(
        model             = model,
        args              = training_args,
        train_dataset     = train_dataset,
        data_collator     = data_collator,
    )

    trainer.train()

    # --------------------------------------------------------------- save
    os.makedirs(cfg["output_dir"], exist_ok=True)
    model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"\n✅ Model saved to: {cfg['output_dir']}")

    # ----------------------------------------------------------- evaluate
    evaluate(model, tokenizer, label_map, cfg)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    args = parser.parse_args()
    main(args.config)
