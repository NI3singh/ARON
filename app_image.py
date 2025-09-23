import gradio as gr
import torch
from diffusers import FluxPipeline

# --- Configuration ---
MODEL_PATH = "./models/flux-1-dev"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 # FLUX works best with bfloat16
print(f"--- Loading FLUX.1 Model on {DEVICE} ---")

# --- Load the FLUX Pipeline ---
# This single pipeline object contains all the necessary model components
try:
    pipe = FluxPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=DTYPE
    ).to(DEVICE)
    print("--- Model Loaded Successfully ---")
except Exception as e:
    print(f"--- Failed to load model: {e} ---")
    pipe = None

# --- Gradio Image Generation Function ---
def generate_image(prompt):
    if pipe is None:
        raise gr.Error("The model pipeline failed to load. Please check the server logs.")

    # Let the user know we're working
    print(f"Generating image for prompt: '{prompt}'")

    # Generate the image. The output is an object containing the image(s).
    # We access the first image with .images[0]
    image = pipe(prompt=prompt, num_inference_steps=28, guidance_scale=7.5).images[0]
    
    print("Image generation complete.")
    return image

# --- Build the Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# 🎨 FLUX.1 Image Generator")
    gr.Markdown("Enter a prompt to create an image using the FLUX.1-dev model.")
    
    with gr.Row():
        prompt_input = gr.Textbox(label="Prompt", placeholder="e.g., A majestic lion wearing a crown, cinematic photo", scale=4)
        submit_button = gr.Button("Generate", scale=1)
    
    output_image = gr.Image(label="Result")
    
    # Connect the button click to the generation function
    submit_button.click(
        fn=generate_image,
        inputs=prompt_input,
        outputs=output_image
    )

# --- Launch the App ---
if __name__ == "__main__":
    # The share=True flag creates a public link to the UI
    demo.launch(share=True)