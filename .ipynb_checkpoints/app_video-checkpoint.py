import gradio as gr
import subprocess
import os
from pathlib import Path

# --- Configuration ---
# Path to the directory where you cloned the Wan2.2 repository
WAN22_REPO_PATH = "./Wan2.2"
# Path to the downloaded model checkpoints
MODEL_CKPT_PATH = "../models/Wan2.2-T2V-A14B" # Note: relative path from inside the repo

print("--- Wan2.2 T2V Gradio Interface ---")
print(f"Using Wan2.2 repository at: {Path(WAN22_REPO_PATH).resolve()}")
print(f"Using Model checkpoints at: {Path(MODEL_CKPT_PATH).resolve(strict=False)}")


# --- Gradio Video Generation Function ---
def generate_video(prompt):
    print(f"\nReceived prompt: '{prompt}'")
    print("Executing Wan2.2 generation script... This will take several minutes.")

    # We construct the exact command-line instruction the model's authors provided.
    # We use --offload_model and --convert_model_dtype to save VRAM on a single GPU setup.
    command = [
        "python", "generate.py",
        "--task", "t2v-A14B",
        "--size", "1280*720",
        "--ckpt_dir", MODEL_CKPT_PATH,
        "--prompt", prompt,
        "--offload_model", "True",
        "--convert_model_dtype",
    ]

    try:
        # We run the command from inside the Wan2.2 repository directory
        # The 'capture_output=True' and 'text=True' flags let us see the output and errors.
        result = subprocess.run(
            command,
            cwd=WAN22_REPO_PATH,
            check=True,
            capture_output=True,
            text=True
        )
        print("--- Script execution successful ---")
        print("Script output:\n", result.stdout)

        # The generate.py script saves videos to its 'output' folder.
        # We need to find the latest video file created there.
        output_dir = Path(WAN22_REPO_PATH) / "output"
        
        # Find all .mp4 files and get the most recently created one
        video_files = sorted(output_dir.glob("**/*.mp4"), key=os.path.getctime, reverse=True)
        
        if not video_files:
            raise gr.Error("Video generation script ran, but no MP4 file was found in the output directory.")
            
        latest_video_path = video_files[0]
        print(f"Found generated video: {latest_video_path}")
        return str(latest_video_path)

    except subprocess.CalledProcessError as e:
        print("!!! --- SCRIPT EXECUTION FAILED --- !!!")
        print("Return Code:", e.returncode)
        print("Error Output:\n", e.stderr)
        # Raise a Gradio error to display the failure message in the UI
        raise gr.Error(f"The model script failed. Check the terminal for error details. Error message: {e.stderr[-500:]}") # Show last 500 chars of error
    except Exception as e:
        raise gr.Error(f"An unexpected error occurred: {e}")


# --- Build the Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# 🎥 Wan2.2 Text-to-Video Generator")
    gr.Markdown("Enter a prompt to create a video using the Wan2.2-T2V-A14B model. Generation can take several minutes.")
    
    with gr.Row():
        prompt_input = gr.Textbox(label="Prompt", placeholder="e.g., A majestic lion wearing a crown, cinematic video", scale=4)
        submit_button = gr.Button("Generate Video", scale=1)
    
    output_video = gr.Video(label="Result")
    
    submit_button.click(
        fn=generate_video,
        inputs=prompt_input,
        outputs=output_video
    )

# --- Launch the App ---
if __name__ == "__main__":
    demo.launch(share=True)