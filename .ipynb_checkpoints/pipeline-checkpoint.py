import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

# --- Configuration ---
MODEL_PATH = "./models/llama-3.1-8b-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Loading Llama 3.1 on {DEVICE} ---")

# --- Load Model and Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE,
)
print("--- Model Loaded Successfully ---")


# --- The Master Prompt ---
# This is where we instruct the LLM on its role and the exact output format we need.
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

def generate_storyboard(topic):
    print(f"\n--- Generating storyboard for topic: '{topic}' ---")
    
    # Format the final prompt
    final_prompt = master_prompt_template.format(user_topic=topic)
    
    messages = [
        { "role": "user", "content": final_prompt }
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(DEVICE)

    # Generate the response from the LLM
    outputs = model.generate(
        input_ids,
        max_new_tokens=1024,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    
    # Decode the response and clean it up
    response_text = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    
    print("--- Raw LLM Output ---")
    print(response_text)
    
    # --- Try to parse the JSON ---
    try:
        # Clean up any markdown code block formatting
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
            
        storyboard_json = json.loads(response_text)
        print("\n--- Successfully Parsed Storyboard ---")
        # Pretty print the JSON
        print(json.dumps(storyboard_json, indent=2))
        return storyboard_json
        
    except Exception as e:
        print(f"\n--- FAILED to parse JSON from LLM output. Error: {e} ---")
        return None

# --- Run the Test ---
if __name__ == "__main__":
    test_topic = "A brave cat astronaut exploring a mysterious alien planet."
    generate_storyboard(test_topic)