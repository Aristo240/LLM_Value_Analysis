# src/smoke_test.py - Now a smoke test to confirm environment health.

import torch
from transformers import pipeline

def check_system():
    print("--- SYSTEM CHECK ---")

    # 1. Determine the device (GPU or CPU)
    # If Cuda is available, use the GPU (cuda:0 is the first GPU). Otherwise, use the CPU.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_status = "GPU (Cuda)" if device.type == "cuda" else "CPU (Slow)"
    
    print(f"Compute Device Available: {device_status}")
    print(f"Pipeline running on: {device}")

    # --- 2. Loading a Tiny Model and FORCING it onto the GPU ---
    print("\nDownloading/Loading Model (gpt2) and moving to GPU...")
    
    # We pass the 'device' object directly to the pipeline to force GPU usage.
    generator = pipeline('text-generation', model='gpt2', device=device)
    
    # 3. Run a simple prompt
    prompt = "The most important human value is"
    print(f"\nPrompting model with: '{prompt}'")
    
    result = generator(prompt, max_length=30, num_return_sequences=1)
    
    print("\n--- MODEL OUTPUT ---")
    print(result[0]['generated_text'])
    print("--------------------")
    print("SUCCESS: System is ready for high-speed research.")

if __name__ == "__main__":
    check_system()