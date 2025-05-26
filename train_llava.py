import argparse
import os
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer
from llava.model.builder import load_pretrained_model # For loading the base LLaVA model
from llava_dataset import tokenize_and_create_labels_for_llava # Custom collate function

def main(args):
    """
    Main function to fine-tune a LLaVA model with LoRA.
    """
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model, Tokenizer, and Image Processor
    print(f"Loading base LLaVA model: {args.model_id}...")
    try:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=args.model_id,
            model_base=None, # LLaVA 1.6 is a full model, not just language adapters
            model_name=args.model_id.split('/')[-1], # Extract model name from path
            load_4bit=args.load_in_4bit,
            # device_map='auto' # Can be useful for multi-GPU, but ensure compatibility
            device=device # Explicitly set device if not using device_map
        )
        print("Base model, tokenizer, and image processor loaded successfully.")
    except Exception as e:
        print(f"Error loading base model: {e}")
        return

    # LoRA Configuration
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules.split(','), # Expect comma-separated string
        lora_dropout=args.lora_dropout,
        bias="none" # As specified
    )
    model = get_peft_model(model, lora_config)
    print("LoRA configured and applied to the model.")
    model.print_trainable_parameters()

    # Load and Prepare Dataset
    print(f"Loading dataset from {args.data_path}...")
    try:
        # Assuming data_path points to a single JSON file as prepared by LLaVA scripts
        full_dataset = load_dataset('json', data_files=args.data_path, split='train')
    except Exception as e:
        print(f"Failed to load dataset from {args.data_path}: {e}")
        return

    if len(full_dataset) == 0:
        print(f"Error: Dataset at {args.data_path} is empty.")
        return

    # Train/eval split
    if args.eval_split_ratio > 0:
        split_dataset = full_dataset.train_test_split(test_size=args.eval_split_ratio, shuffle=True, seed=args.seed)
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']
        print(f"Dataset split: {len(train_dataset)} training samples, {len(eval_dataset)} evaluation samples.")
    else:
        train_dataset = full_dataset
        eval_dataset = None
        print(f"Using full dataset for training: {len(train_dataset)} samples. No evaluation split.")


    # Data Collator
    # The collate_fn needs access to tokenizer, image_processor, model.config, and image_dir
    # We can wrap it or pass these fixed arguments using a lambda or functools.partial
    # model.config is crucial for process_images within the collator
    # Note: The tokenizer and image_processor are already loaded.
    
    # Make sure model.config.torch_dtype is set if not already (often done by load_pretrained_model)
    if not hasattr(model.config, 'torch_dtype'):
        model.config.torch_dtype = torch.float16 if args.load_in_4bit else torch.float32


    collate_fn_wrapper = lambda batch: tokenize_and_create_labels_for_llava(
        batch=batch,
        tokenizer=tokenizer,
        image_processor=image_processor,
        model_config=model.config, # Pass the model's config
        image_dir=args.image_dir,
        ignore_index=-100, # Standard ignore index for Hugging Face
        device=str(device) # Pass device as string
    )

    # TrainingArguments
    print("Setting up TrainingArguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size if eval_dataset else args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        save_strategy="steps" if eval_dataset else "epoch", # Save by steps if evaluating, else by epoch
        save_steps=args.save_steps if eval_dataset and args.save_steps > 0 else 0.2, # Fraction of total steps per epoch, or integer
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=args.eval_steps if eval_dataset and args.eval_steps > 0 else 0.2, # Fraction of total steps per epoch, or integer
        logging_steps=args.logging_steps,
        bf16=args.bf16 and torch.cuda.is_bf16_supported(),
        fp16=not args.bf16 and args.fp16 and torch.cuda.is_available(), # Use fp16 if bf16 not chosen and fp16 is
        dataloader_pin_memory=True, # Can be beneficial
        save_total_limit=3,
        remove_unused_columns=False, # Important for custom collate_fn
        push_to_hub=False,
        label_names=["labels"],
        load_best_model_at_end=True if eval_dataset else False,
        report_to=None, # "wandb", "tensorboard"
        optim="adamw_torch",
        ddp_find_unused_parameters=False, # Often needed for LLaVA DDP
        seed=args.seed,
    )

    # Initialize Trainer
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn_wrapper,
    )

    # Train
    print("Starting fine-tuning...")
    try:
        trainer.train()
        print("Fine-tuning completed.")
    except Exception as e:
        print(f"Error during training: {e}")
        # Consider saving the current state if an error occurs
        # trainer.save_model(os.path.join(args.output_dir, "error_checkpoint"))
        # print(f"Saved error checkpoint to {os.path.join(args.output_dir, 'error_checkpoint')}")
        return

    # Save the final model (LoRA adapters)
    print(f"Saving fine-tuned LoRA adapters to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    # The tokenizer is typically saved with the base model, but if you made custom changes
    # (e.g., added special tokens not part of the base), save it too.
    # tokenizer.save_pretrained(args.output_dir)
    print("Model saving completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a LLaVA model with LoRA.")
    
    parser.add_argument("--model_id", type=str, default="liuhaotian/llava-v1.6-mistral-7b",
                        help="Hugging Face model ID for the base LLaVA model.")
    parser.add_argument("--data_path", type=str, default="final_data.json",
                        help="Path to the JSON data file for fine-tuning.")
    parser.add_argument("--image_dir", type=str, default="images",
                        help="Directory containing the images referenced in the data file.")
    parser.add_argument("--output_dir", type=str, default="llava-v1.6-mistral-7b-finetuned-med",
                        help="Directory to save the fine-tuned LoRA adapters.")
    
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device training batch size (adjust based on VRAM).")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Per-device evaluation batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.") # LLaVA often uses smaller LR
    
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA r parameter.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate.")
    parser.add_argument("--lora_target_modules", type=str, 
                        default="q_proj,k_proj,v_proj,o_proj,mm_projector,up_proj,down_proj,gate_proj", # Common LLaVA targets
                        help="Comma-separated list of LoRA target modules.")

    parser.add_argument("--load_in_4bit", action='store_true', default=True,
                        help="Load the base model in 4-bit precision.")
    parser.add_argument("--no_load_in_4bit", action='store_false', dest='load_in_4bit',
                        help="Do not load the base model in 4-bit precision.")
                        
    parser.add_argument("--bf16", action='store_true', default=torch.cuda.is_bf16_supported(),
                        help="Use bfloat16 mixed precision if available. Overrides fp16 if both specified.")
    parser.add_argument("--fp16", action='store_true', default=False,
                        help="Use float16 mixed precision if bf16 is not used.")

    parser.add_argument("--eval_split_ratio", type=float, default=0.1,
                        help="Ratio of the dataset to use for evaluation (e.g., 0.1 for 10%). Set to 0 for no evaluation.")
    parser.add_argument("--save_steps", type=float, default=0.2,
                        help="Save checkpoint every X fraction of an epoch's steps, or as an integer number of steps if > 1.")
    parser.add_argument("--eval_steps", type=float, default=0.2,
                        help="Evaluate every X fraction of an epoch's steps, or as an integer number of steps if > 1.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()
    
    # Adjust steps if they are intended as fractions
    if args.save_steps <= 1.0 and args.save_steps > 0: # Treat as fraction
        # This calculation needs total number of steps, which depends on dataset length
        # For now, let Trainer handle it if it supports float for steps as fraction of epoch steps.
        # Typically, Trainer expects integer step counts. This logic might need refinement or
        # for user to provide integer steps based on their dataset.
        print(f"Note: save_steps={args.save_steps} as a fraction. Trainer might interpret this as total steps if it's an integer.")
    if args.eval_steps <= 1.0 and args.eval_steps > 0:
         print(f"Note: eval_steps={args.eval_steps} as a fraction. Trainer might interpret this as total steps if it's an integer.")


    main(args)
