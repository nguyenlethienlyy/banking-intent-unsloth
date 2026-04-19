# =============================================================================
# inference.py
# Standalone inference module for banking intent classification.
#
# Usage (script):
#   python scripts/inference.py --config configs/inference.yaml
#
# Usage (import):
#   from scripts.inference import IntentClassification
#   model = IntentClassification("configs/inference.yaml")
#   print(model("I lost my credit card"))
# =============================================================================

import argparse
import json

import torch
import yaml
from unsloth import FastLanguageModel


# =============================================================================
# Prompt template  (must match train.py exactly)
# =============================================================================

PROMPT_TEMPLATE = """### Instruction:
Classify the banking query below into one of the intent categories.

### Input:
{text}

### Response:
"""


# =============================================================================
# IntentClassification class  (interface required by the assignment)
# =============================================================================

class IntentClassification:

    def __init__(self, model_path: str):
        """
        Load config, tokenizer, and model checkpoint.

        Args:
            model_path: path to inference.yaml config file.
        """
        # --------------------------------------------------------- config
        with open(model_path, "r") as f:
            cfg = yaml.safe_load(f)

        # --------------------------------------------------------- label map
        with open(cfg["label_map_path"], "r") as f:
            label_map = json.load(f)

        # id → intent name
        self.id_to_label = {v: k for k, v in label_map.items()}
        self.label_to_id = label_map
        self.known_labels = list(label_map.keys())

        # --------------------------------------------------------- model
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name     = cfg["model_checkpoint"],
            max_seq_length = cfg["max_seq_length"],
            load_in_4bit   = cfg["load_in_4bit"],
            dtype          = None,
        )
        FastLanguageModel.for_inference(self.model)

    # ------------------------------------------------------------------

    def __call__(self, message: str) -> str:
        """
        Predict the intent label for a single input message.

        Args:
            message: raw user query string.

        Returns:
            predicted_label: intent name string (e.g. 'lost_or_stolen_card').
        """
        prompt = PROMPT_TEMPLATE.format(text=message.strip())

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens = 20,
                pad_token_id   = self.tokenizer.eos_token_id,
                do_sample      = False,
            )

        decoded  = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = decoded.split("### Response:")[-1].strip().lower()

        # match response text against known label names
        predicted_label = "unknown"
        for label_name in self.known_labels:
            if label_name.lower() in response:
                predicted_label = label_name
                break

        return predicted_label


# =============================================================================
# Entry point  (demo run)
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/inference.yaml")
    args = parser.parse_args()

    print("Loading model ...")
    classifier = IntentClassification(args.config)

    # --- demo examples ---
    examples = [
        "I lost my credit card",
        "Why hasn't my balance updated after the transfer?",
        "How do I change my PIN?",
        "My card is not working at the ATM",
        "I want to cancel my recent transfer",
    ]

    print("\n--- Inference Demo ---")
    for text in examples:
        label = classifier(text)
        print(f"  Input : {text}")
        print(f"  Intent: {label}\n")
