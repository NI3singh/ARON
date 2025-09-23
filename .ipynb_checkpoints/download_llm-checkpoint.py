import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

# --- Configuration ---
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SAVE_PATH = Path("./models/llama-3.1-8b-instruct")

print(f"--- Starting download for: {MODEL_ID} ---")
print(f"Model will be saved to: {SAVE_PATH.resolve()}")

# Create the save directory if it doesn't exist
SAVE_PATH.mkdir(parents=True, exist_ok=True)

# --- Download and Save ---
try:
    # Download tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 # Use bfloat16 for better performance
    )

    # Save them to the local directory
    print(f"\nSaving model and tokenizer to {SAVE_PATH}...")
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)

    print(f"\n✅ Successfully downloaded and saved model.")

except Exception as e:
    print(f"\n❌ An error occurred: {e}")