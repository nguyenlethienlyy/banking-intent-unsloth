# =============================================================================
# preprocess_data.py
# Loads a 10-intent subset of BANKING77, cleans text, maps labels,
# and saves train.csv / test.csv / label_map.json to sample_data/.
#
# No CLI arguments — edit the CONFIG section below.
# =============================================================================

# =============================================================================
# CONFIG
# =============================================================================

OUTPUT_DIR            = "sample_data"   # output folder for CSV and label map
TEST_SIZE             = 0.2             # proportion of data reserved for test
RANDOM_STATE          = 42              # random seed for reproducibility
MAX_SAMPLES_PER_CLASS = 200             # max rows per intent class (balancing)

SELECTED_INTENTS = [
    "balance_not_updated_after_bank_transfer",
    "beneficiary_not_allowed",
    "cancel_transfer",
    "card_arrival",
    "card_linking",
    "card_not_working",
    "card_payment_fee_charged",
    "change_pin",
    "lost_or_stolen_card",
    "top_up_failed",
]

# =============================================================================
# Imports
# =============================================================================

import os
import re
import json

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


# =============================================================================
# Helpers
# =============================================================================

def clean_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip("\"'")
    return text


def build_label_map(intents: list) -> dict:
    """Deterministic intent → integer mapping (alphabetical order)."""
    return {intent: idx for idx, intent in enumerate(sorted(intents))}


# =============================================================================
# Main pipeline
# =============================================================================

def preprocess():
    print("=" * 60)
    print("  BANKING77 Preprocessing")
    print("=" * 60)

    # ------------------------------------------------------------------ load
    print("\n[1/5] Loading BANKING77 from HuggingFace ...")
    dataset = load_dataset("banking77")
    label_names = dataset["train"].features["label"].names  # 77 intent strings
    original_id_to_name = {i: name for i, name in enumerate(label_names)}

    # ----------------------------------------------------------------- filter
    print(f"[2/5] Filtering to {len(SELECTED_INTENTS)} selected intents ...")

    def keep_row(example):
        return original_id_to_name[example["label"]] in SELECTED_INTENTS

    filtered_train = dataset["train"].filter(keep_row)
    filtered_test  = dataset["test"].filter(keep_row)

    # merge official splits so we can re-split ourselves
    all_texts  = list(filtered_train["text"])  + list(filtered_test["text"])
    all_labels = list(filtered_train["label"]) + list(filtered_test["label"])

    # ------------------------------------------------------------------ clean
    print("[3/5] Cleaning and normalising text ...")
    all_texts        = [clean_text(t) for t in all_texts]
    all_intent_names = [original_id_to_name[l] for l in all_labels]

    label_map     = build_label_map(SELECTED_INTENTS)
    all_label_ids = [label_map[name] for name in all_intent_names]

    df = pd.DataFrame({
        "text":   all_texts,
        "intent": all_intent_names,
        "label":  all_label_ids,
    })

    # ---------------------------------------------------------------- balance
    print(f"[4/5] Balancing — capping at {MAX_SAMPLES_PER_CLASS} per class ...")
    df = (
        df.groupby("intent", group_keys=False)
          .apply(lambda g: g.sample(min(len(g), MAX_SAMPLES_PER_CLASS),
                                    random_state=RANDOM_STATE))
          .reset_index(drop=True)
    )

    print(f"      Total samples : {len(df)}")
    print(f"      Per intent    :\n{df['intent'].value_counts().to_string()}\n")

    # ------------------------------------------------------------------ split
    print("[5/5] Splitting into train / test sets ...")
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    print(f"      Train : {len(train_df)} samples")
    print(f"      Test  : {len(test_df)} samples")

    # ------------------------------------------------------------------- save
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_path     = os.path.join(OUTPUT_DIR, "train.csv")
    test_path      = os.path.join(OUTPUT_DIR, "test.csv")
    label_map_path = os.path.join(OUTPUT_DIR, "label_map.json")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,   index=False)

    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\n  Saved: {train_path}")
    print(f"  Saved: {test_path}")
    print(f"  Saved: {label_map_path}")

    # --------------------------------------------------------------- sanity
    assert not train_df["text"].isnull().any(), "Null values in train text!"
    assert not test_df["text"].isnull().any(),  "Null values in test text!"
    assert set(train_df["label"].unique()) == set(test_df["label"].unique()), \
        "Label mismatch between train and test!"

    print("\n✅ Preprocessing complete!")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    preprocess()
