import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import FluxPipeline
import json
import time
import subprocess # --- NEW ---
from pathlib import Path # --- NEW ---
import os # --- NEW ---

# --- Configuration ---
LLM_PATH = "./models/llama-3.1-8b-instruct"
IMAGE_MODEL_PATH = "./models/flux-1-dev"
WAN22_REPO_PATH = "./Wan2.2" # --- NEW ---
WAN22_MODEL_CKPT_PATH = "../models/Wan2.2-T2V-A14B" # --- NEW ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# --- 1. Load All Models ---
print(f"--- Loading all models on {DEVICE} ---")
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
llm_model = AutoModelForCausalLM.from_pretrained(LLM_PATH, torch_dtype=DTYPE, device_map=DEVICE)
print("--- Llama 3.1 LLM Loaded Successfully ---")

try:
    image_pipe = FluxPipeline.from_pretrained(IMAGE_MODEL_PATH, torch_dtype=DTYPE).to(DEVICE)
    print("--- FLUX.1 Image Model Loaded Successfully ---")
except Exception as e:
    print(f"--- Failed to load FLUX.1 model: {e} ---")
    image_pipe = None

print("--- All Models Loaded ---")


# --- 2. Storyboard & Image Generation Logic (No Changes) ---
def generate_storyboard(user_topic):
    print(f"\n--- Received topic: '{user_topic}' ---")
    master_prompt_template = """
    You are an expert AI Video Director. Your task is to take a user's topic and create a storyboard for a short 15-second video.
    The storyboard must be broken down into exactly 3 distinct scenes.
    Your final output must be ONLY a valid JSON object, with no other text before or after it.
    The JSON object must contain a single key "storyboard" which is a list of scene objects.
    Each scene object in the list must have two keys:
    1. "scene_description": A short, one-sentence description of the scene's action.
    2. "visual_prompt": A detailed, visually rich prompt for an AI image generator, including style keywords like 'cinematic, photorealistic, 4k'.

    User's Topic: "{user_topic}"
    """
    final_prompt = master_prompt_template.format(user_topic=user_topic)
    messages = [{"role": "user", "content": final_prompt}]
    input_ids = llm_tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(DEVICE)
    outputs = llm_model.generate(input_ids, max_new_tokens=1024, eos_token_id=llm_tokenizer.eos_token_id, do_sample=True, temperature=0.7, top_p=0.9)
    response_text = llm_tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
    try:
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        storyboard_json = json.loads(response_text)
        print("--- Storyboard generated successfully. ---")
        return storyboard_json, storyboard_json
    except Exception as e:
        raise gr.Error(f"Failed to generate a valid storyboard. Raw output: {response_text}")

def generate_images_from_storyboard(storyboard_data):
    if image_pipe is None: raise gr.Error("Image model failed to load.")
    if not storyboard_data or "storyboard" not in storyboard_data: raise gr.Error("Generate a storyboard first.")
    print(f"\n--- Generating images for {len(storyboard_data['storyboard'])} scenes ---")
    images = []
    for i, scene in enumerate(storyboard_data["storyboard"]):
        prompt = scene.get("visual_prompt", "")
        print(f"Scene {i+1} Prompt: '{prompt}'")
        image = image_pipe(prompt=prompt, num_inference_steps=28, guidance_scale=7.5).images[0]
        images.append(image)
    print("--- All images generated successfully ---")
    return images, images


# --- NEW: Video Generation Logic ---
# --- NEW (CORRECTED): Video Generation Logic ---
def generate_videos_from_images(storyboard_data, image_list):
    if not image_list: raise gr.Error("Please generate images first.")
    
    print(f"\n--- Generating videos for {len(image_list)} images ---")
    
    # Create a temporary directory for image inputs
    temp_image_dir = Path("./temp_images")
    temp_image_dir.mkdir(exist_ok=True)

    video_outputs = []
    for i, (scene, image) in enumerate(zip(storyboard_data["storyboard"], image_list)):
        print(f"--- Processing Scene {i+1} ---")
        
        # 1. Save the PIL image to a temporary file
        temp_image_path = temp_image_dir / f"scene_{i+1}.png"
        image.save(temp_image_path)
        print(f"Saved temporary image to {temp_image_path}")

        # 2. Get the prompt from the storyboard
        prompt = scene.get("visual_prompt", "A beautiful cinematic video.")

        # 3. Construct and run the command for Wan2.2's generate.py
        command = [
            "python", "generate.py",
            "--task", "i2v-A14B", # Image-to-Video task
            
            # --- THE FIX IS HERE ---
            "--image", str(temp_image_path.resolve()), # Corrected from --image_path
            
            "--ckpt_dir", WAN22_MODEL_CKPT_PATH,
            "--prompt", prompt,
            "--offload_model", "True",
            "--convert_model_dtype"
        ]
        
        try:
            print(f"Running subprocess for Scene {i+1}... This will take several minutes.")
            subprocess.run(command, cwd=WAN22_REPO_PATH, check=True, capture_output=True, text=True)
            
            # 4. Find the generated video
            output_dir = Path(WAN22_REPO_PATH) / "output"
            video_files = sorted(output_dir.glob("**/*.mp4"), key=os.path.getctime, reverse=True)
            if video_files:
                video_outputs.append(str(video_files[0]))
                print(f"--- Successfully generated video for Scene {i+1} ---")
            else:
                print(f"--- WARNING: Video script ran, but no MP4 found for Scene {i+1} ---")

        except subprocess.CalledProcessError as e:
            print(f"!!! --- VIDEO SCRIPT FAILED for Scene {i+1} --- !!!")
            print("Error:", e.stderr)
            gr.Warning(f"Failed to generate video for Scene {i+1}. Check terminal for details.")
            continue # Continue to the next scene

    print("--- All video generation attempts complete. ---")
    return video_outputs

# --- 3. Build the Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft(), title="AI Video Creator") as demo:
    gr.Markdown("# 🎬 AI Video Creator Prototype")
    
    storyboard_state = gr.State(value=None)
    image_state = gr.State(value=None) # --- NEW ---
    
    with gr.Tabs():
        with gr.TabItem("🏠 Welcome"):
            gr.Markdown("## Welcome! Navigate through the tabs to generate a video from a single idea.")
        
        with gr.TabItem("📝 Step 1: Storyboard"):
            # ... (No changes from before)
            topic_input = gr.Textbox(label="Your Video Idea", placeholder="e.g., A brave cat astronaut exploring a mysterious alien planet.")
            generate_story_btn = gr.Button("Generate Storyboard", variant="primary")
            storyboard_output = gr.JSON(label="Generated Storyboard")
            generate_story_btn.click(fn=generate_storyboard, inputs=topic_input, outputs=[storyboard_output, storyboard_state])

        with gr.TabItem("🎨 Step 2: Images"):
            # ... (UPDATED to pass images to the new state)
            generate_imgs_btn = gr.Button("Generate Images from Storyboard", variant="primary")
            storyboard_display = gr.JSON(label="Current Storyboard")
            gallery_output = gr.Gallery(label="Scene Images", columns=3, height="auto")
            demo.load(lambda x: x, inputs=storyboard_state, outputs=storyboard_display)
            generate_imgs_btn.click(fn=generate_images_from_storyboard, inputs=storyboard_state, outputs=[gallery_output, image_state])

        # --- NEW: The entire Video Generator Tab ---
        with gr.TabItem("🎥 Step 3: Videos"):
            gr.Markdown("## Step 3: Generate Video Clips")
            gr.Markdown("Click the button below to animate each of the generated images. This is the slowest step and may take several minutes per scene.")
            generate_vids_btn = gr.Button("Generate Videos from Images", variant="primary")
            
            gr.Markdown("### Source Images:")
            image_display = gr.Gallery(label="Scene Images", columns=3, height="auto")
            
            gr.Markdown("### Generated Video Clips:")
            video_gallery_output = gr.Gallery(label="Scene Videos", columns=3, height="auto")

            demo.load(lambda x: x, inputs=image_state, outputs=image_display)
            generate_vids_btn.click(fn=generate_videos_from_images, inputs=[storyboard_state, image_state], outputs=video_gallery_output)

# --- 4. Launch the App ---
if __name__ == "__main__":
    demo.launch(share=True)