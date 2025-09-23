import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread

# --- Configuration ---
MODEL_PATH = "./models/llama-3.1-8b-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Loading Model on {DEVICE} ---")

# --- Load Model and Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE,
)
print("--- Model Loaded Successfully ---")

# --- Gradio Chat Function ---
def chat_function(message, history):
    # System prompt can be added here if needed
    # For now, we'll keep it a simple chat
    
    # Format the conversation history for the model
    conversation = []
    for user_msg, assistant_msg in history:
        conversation.append({"role": "user", "content": user_msg})
        conversation.append({"role": "assistant", "content": assistant_msg})
    conversation.append({"role": "user", "content": message})
    
    # Tokenize the input
    input_ids = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(DEVICE)

    # Use a streamer for real-time text generation
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(input_ids=input_ids, streamer=streamer, max_new_tokens=512)
    
    # Run generation in a separate thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Yield the generated text as it comes
    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        yield generated_text

# --- Build the Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Director LLM Chat")
    gr.Markdown("Chat directly with the Llama 3.1 Instruct model.")
    gr.ChatInterface(chat_function)

# --- Launch the App ---
if __name__ == "__main__":
    # The share=True flag will create a public link that you can use to access the UI from your local browser.
    demo.launch(share=True)