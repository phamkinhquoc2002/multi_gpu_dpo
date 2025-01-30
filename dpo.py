import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset,Dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
from accelerate import PartialState
import time
from datetime import datetime
from typing import Dict

def chat_format(example) -> Dict:

    prompt = "<|im_start|>user\n" + generation_prompt + example['full_karteText'] + "<|im_end|>\n<|im_start|>assistant\n"
    chosen = example['tsdata'] + "<|im_end|>\n"
    rejected = example['rejected'] + "<|im_end|>\n"

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }

def data_preparation(
    config: Dict
) -> Dataset:
    dataset = load_dataset(
            'csv', data_files=config['dataset_name'], split='train')
    dataset =  dataset.map(chat_format, remove_columns=dataset.column_names)
    dataset =  dataset.train_test_split(test_size=0.2)
    return dataset


def main():
    # Configuration Load
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    # DDP - Data Parallelism Configuration
    device_map = 'DDP'
    if device_map == 'DDP':
        device_string = PartialState().process_index
        device_map = {'': device_string}
    # Dataset Load
    dataset = data_preparation(config=config)
    prompt = config['generation_prompt']
    # Model Load
    model_name = config['training_config']['model_name']
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir="./models",
        use_cache=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device_map
    ) 
    # Prepare the model for Lora
    model = prepare_model_for_kbit_training(base_model)
    model.to(torch.bfloat16)
    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding="left", model_max_length=config['training_config']['max_length'], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    #Lora Configuration
    lora_config = LoraConfig(
        #Lora Rank
        r=config['training_config']['peft_config']['lora_r'],
        lora_alpha=config['training_config']['peft_config']['lora_alpha'],
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=config['training_config']['peft_config']['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM",
    )

    dpo_config = DPOConfig(
        output_dir="./results",
        #1つのGPUあたりのバッチサイズ。2つのGPUがある場合、1回の処理で2倍のバッチサイズを処理します。
        per_device_train_batch_size=config['training_config']['dpo_config']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training_config']['dpo_config']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training_config']['dpo_config']['gradient_accumulation_steps'],
        ##最適化関数、別のものに変更可能。詳しくはhuggingface.comをご確認ください！
        optim='adamw_8bit',
        save_steps=2,
        learning_rate=2e-4,
        bf16 = True,
        fp16= False,
        max_grad_norm=config['training_config']['dpo_config']['max_grad_norm'],
        logging_steps=1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_completion_length=config['training_config']['max_completion_length'],
        num_train_epochs=config['training_config']['dpo_config']['num_train_epochs'],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="wandb"
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        peft_config=lora_config,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        processing_class=tokenizer,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {trainable_params}")
    start_time = time.time()
    try:
        trainer.train()
    except Exception as e:
        print(f"Training error: {e}")
        trainer.save_model("./emergency_checkpoint")
    finally:
        # Track time: End time
        end_time = time.time()
        time_run = end_time - start_time
        trainer.model.save_pretrained("final_checkpoint")
        tokenizer.save_pretrained("final_checkpoint")
        # Calculate total time spent on training (in seconds)
        time_run = end_time - start_time
        model = PeftModel.from_pretrained(base_model, "final_checkpoint")
        model = model.merge_and_unload()
        model.push_to_hub(config['save_config']['new_model'], use_temp_dir=False, token=config['save_config']['hf_token'])
        tokenizer.push_to_hub(config['save_config']['new_model'], use_temp_dir=False, token=config['save_config']['hf_token'])
        with open(f'training_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.yaml', 'w') as f:
            yaml.dump({
                'time_run': time_run,
                'cost': (time_run / 3600) * config['gpu_config']['hourly_cost'],
            }, f)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
