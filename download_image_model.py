import torch
from diffusers import DiffusionPipeline, FluxPipeline
from pathlib import Path

# --- Configuration ---
MODEL_ID = "black-forest-labs/FLUX.1-dev"
SAVE_PATH = Path("./models/flux-1-dev")

print(f"--- Starting download for Image Model: {MODEL_ID} ---")
print(f"Model will be saved to: {SAVE_PATH.resolve()}")

# Create the save directory if it doesn't exist
SAVE_PATH.mkdir(parents=True, exist_ok=True)

# --- Download and Save ---
try:
    # FLUX uses a specific pipeline class called FluxPipeline
    # We load the weights in bfloat16 for efficiency
    pipeline = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16
    )

    # Save the entire pipeline to our local directory
    print(f"\nSaving pipeline to {SAVE_PATH}...")
    pipeline.save_pretrained(SAVE_PATH)

    print(f"\n✅ Successfully downloaded and saved the FLUX.1 image model.")

except Exception as e:
    print(f"\n❌ An error occurred: {e}")