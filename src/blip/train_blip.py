import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle

from transformers import (
    BlipProcessor,
    BlipForQuestionAnswering,
    AdamW,
    get_scheduler
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Import the custom dataset class
from .blip_dataset import VQADataset

def main(args):
    """
    Main function to fine-tune the BLIP VQA model.
    """
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Processor and Model
    print("Loading BLIP processor and model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    model.to(device)

    # LoRA Configuration
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        target_modules=["query", "value"], # As in the notebook
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        # task_type=TaskType.SEQ_2_SEQ_LM # Not specified for BLIP VQA in notebook, might not be needed
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and Prepare Dataset
    print(f"Loading dataset from {args.data_path}...")
    # The notebook uses a split like "train[:20%]", for simplicity here we use the whole 'train' split.
    # Users can preprocess their CSV or adjust this part if specific splits are needed.
    try:
        raw_dataset = load_dataset("csv", data_files=args.data_path, split="train")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Please ensure your CSV file is correctly formatted and the path is correct.")
        return

    print(f"Initializing VQADataset with image directory: {args.image_dir}...")
    train_dataset = VQADataset(dataset_split=raw_dataset,
                               processor=processor,
                               image_dir=args.image_dir)
    
    if len(train_dataset) == 0:
        print("Error: The dataset is empty. Please check data path and image directory.")
        return

    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=args.batch_size, 
                                  shuffle=True, # Shuffle for training
                                  pin_memory=True if device.type == "cuda" else False)

    # Optimizer and Scheduler
    print("Setting up optimizer and scheduler...")
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    # Determine total training steps for scheduler
    num_training_steps = args.epochs * len(train_dataloader)
    
    scheduler = get_scheduler(
        name="exponential_lr", # Corresponds to ExponentialLR in notebook
        optimizer=optimizer,
        # num_warmup_steps=0, # Default in notebook's ExponentialLR
        num_training_steps=num_training_steps,
        # scheduler_specific_kwargs={'gamma': 0.9} # get_scheduler might not take this directly
    )
    # For ExponentialLR, gamma is often part of its direct constructor.
    # Hugging Face's get_scheduler might not expose gamma for 'exponential_lr'.
    # If direct gamma control is needed, using torch.optim.lr_scheduler.ExponentialLR is better.
    # For this conversion, we'll stick to get_scheduler if it works, otherwise, note it.
    # The notebook's scheduler.step() was also commented out in its training loop.
    # We will keep it simple and consistent with the notebook's active parts.
    # If scheduler.step() was intended, it should be called each epoch.

    # Training Loop
    print("Starting training loop...")
    min_train_loss = float("inf")
    early_stopping_hook = 0
    patience = 10  # As in the notebook
    
    # GradScaler for mixed-precision training
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available() and device.type == 'cuda')

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available() and device.type == 'cuda'):
                outputs = model(input_ids=input_ids,
                                pixel_values=pixel_values,
                                attention_mask=attention_mask,
                                labels=labels)
                loss = outputs.loss
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf loss detected at epoch {epoch+1}, batch {batch_idx}. Skipping update.")
                # Potentially add gradient clipping here if this becomes an issue
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                continue


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # scheduler.step() # If using a step-wise scheduler

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} completed. Average Training Loss: {avg_epoch_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")

        # The notebook uses training loss for early stopping as validation is commented out
        if avg_epoch_loss < min_train_loss:
            print(f"Training loss improved from {min_train_loss:.4f} to {avg_epoch_loss:.4f}.")
            min_train_loss = avg_epoch_loss
            early_stopping_hook = 0
            
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            model.save_pretrained(args.output_dir)
            processor.save_pretrained(args.output_dir) # Save processor for easy loading later
            print(f"Saved model and processor to {args.output_dir}")
        else:
            early_stopping_hook += 1
            print(f"Training loss did not improve. Early stopping hook: {early_stopping_hook}/{patience}")
            if early_stopping_hook >= patience:
                print("Early stopping triggered.")
                break
        
        # If scheduler is per-epoch (like ExponentialLR in notebook)
        # scheduler.step() # This was commented out in the notebook's training loop

    # Save tracking information (though it was empty in the notebook)
    # tracking_information = [] # Re-initialize if it was meant to be populated
    # with open(os.path.join(args.output_dir, "training_tracking_information.pkl"), "wb") as f:
    #     pickle.dump(tracking_information, f)
        
    print("Fine-tuning process completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BLIP VQA model with LoRA.")
    
    parser.add_argument("--data_path", type=str, default="train_2.csv",
                        help="Path to the training CSV file.")
    parser.add_argument("--image_dir", type=str, default="figures",
                        help="Directory containing the images referenced in the CSV.")
    parser.add_argument("--output_dir", type=str, default="Model/blip-saved-model",
                        help="Directory to save the fine-tuned model adapters.")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=4e-5,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--lora_r", type=int, default=32,
                        help="LoRA r parameter (rank).")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha parameter.")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout rate.")

    args = parser.parse_args()
    main(args)
