import math

def calculate_vram_requirements(model_params, hidden_size, layer_size, batch_size,seq_length, tuning_option):
    """Calculate the VRAM requirements based on the tuning option."""
    frozen_base_model_size = model_params * 2 / 1e9
    activation_memory = layer_size * (hidden_size * 2 * batch_size * seq_length * 0.6) / 1e9
    if tuning_option == 'quantize-lora':
        # Using quantization
        frozen_base_model_size = frozen_base_model_size / 4
        lora_params = frozen_base_model_size * lora_ratio 
        optimizer_memory = lora_params * 4
        gradient_memory = lora_params
    elif tuning_option == 'lora':
        #Typically, lora_ratio is around 1-2% of original model params
        lora_ratio = 0.02
        lora_params = frozen_base_model_size * lora_ratio 
        optimizer_memory = lora_params * 4
        gradient_memory = lora_params
    elif tuning_option == 'full-finetune':
        # Full fine-tuning in 16-bit
        lora_params = 0
        optimizer_memory = frozen_base_model_size * 4
        gradient_memory = frozen_base_model_size
    else:
        raise ValueError("Invalid tuning option selected.")
    total = optimizer_memory + gradient_memory + activation_memory + lora_params + frozen_base_model_size 
    return total 
    
def main():
    print("Welcome to ファム―さん GPU Configuration Guide!")
    
    model_size_billion_params = float(input("Enter your model size in billions of parameters: "))
    batch_size = int(input("Enter your batch_size: "))
    seq_length = int(input("Enter your sequence maximum length: "))
    hidden_size = int(input("Model hidden dimensions: "))
    layer_size = int(input("Total layers: "))
    tuning_option = input("\nChoose your fine-tuning approach:\n 1 'quantize-lora' <4-bit quantization LoRA> (lowest cost, moderate quality)\n b) 'lora' <16-bit with LoRA> (moderate cost, high quality)\n c) 'full-finetune' <Full fine-tuning in 16-bit> (high cost, highest quality)\nYour choice (a/b/c): ").lower()
    
    estimated_vram = calculate_vram_requirements(model_size_billion_params, hidden_size, layer_size, batch_size, seq_length,tuning_option)
    print(f"\nEstimated VRAM required: ~{estimated_vram:.2f}~ GB. (Not precise!)")
    
    gpu_vram = float(input("Enter amount of VRAM you have per GPU: "))
    min_gpus = estimated_vram / gpu_vram
    print(f"\nMinimum number of GPUs required: {max(1, math.ceil(min_gpus))}")
    
    # GPU setup advice
    if min_gpus <= 1:
        speed_preference = input("Do you wish to maximise speed or minimize GPUs (SPEED, GPUs)? ").lower()
        if speed_preference == 'speed':
            print("\nRecommendation: Use Distributed Data Parallel (DDP) and increase the number of GPUs and batch size for more speed.")
        else:
            print("\nRecommendation: Use Unsloth for minimum-VRAM.")
    else:
        minimize_choice = input("Do you wish to minimize the number of GPUs or training time (GPUs or Finetuning TIME)? ").lower()
        if minimize_choice == 'gpus':
            print("\nRecommendation: Use Unsloth for minimum-VRAM.")
        else:
            print("\nRecommendation: Use Fully Sharded Data Parallel (FSDP) <--coming sooon--> and increase the number of GPUs.")

if __name__ == "__main__":
    main()