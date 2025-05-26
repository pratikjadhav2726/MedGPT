import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class VQADataset(Dataset):
    """
    VQA dataset specialized for medical image QA with choices and a detailed answer format,
    as used in MedGPT_Finetuning.ipynb.
    """

    def __init__(self, dataset_split, processor, image_dir: str):
        """
        Args:
            dataset_split: A split from a Hugging Face Dataset object (e.g., dataset['train']).
                           Each entry is expected to have 'Figure_path', 'Question', 'Caption',
                           'Answer' (label like 'A'), and 'Choice A', 'Choice B', etc.
            processor: The BLIP processor for image and text preprocessing.
            image_dir (str): The directory where images are stored.
        """
        self.processor = processor
        self.image_dir = image_dir
        # The dataset_split is directly used. Filtering for existing images can be done
        # explicitly before creating this Dataset if needed, or handled in __getitem__.
        self.dataset = dataset_split

    def _image_exists(self, image_filename: str) -> bool:
        """Checks if an image file exists in the image_dir."""
        full_path = os.path.join(self.image_dir, image_filename)
        return os.path.exists(full_path)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves and preprocesses a single data sample.
        """
        entry = self.dataset[idx]

        image_filename = entry['Figure_path']
        image_path = os.path.join(self.image_dir, image_filename)

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Image not found at {image_path}. Skipping this sample or returning placeholder.")
            # Depending on desired behavior, you might return None or raise an error.
            # For now, let's try to return a placeholder encoding, or skip by error.
            # This needs careful handling in the DataLoader's collate_fn or by filtering upfront.
            # For simplicity in this conversion, we'll assume images exist or an error is acceptable.
            raise # Re-raise the error to be caught by DataLoader or handled upstream.

        question = entry['Question'].strip()
        caption = entry['Caption'] # Used in answer construction
        answer_label = entry['Answer'] # e.g., 'A', 'B'
        
        # Construct the answer text as done in the notebook
        # Example: entry['Choice A'] might be "A: Golgi complexes"
        # We want "Golgi complexes as, the image is about {caption}"
        choice_key = f'Choice {answer_label}'
        answer_choice_text_with_prefix = entry[choice_key]
        
        # Remove the prefix like "A: " or "B: "
        if ":" in answer_choice_text_with_prefix:
            answer_choice_text = answer_choice_text_with_prefix.split(":", 1)[1].strip()
        else:
            answer_choice_text = answer_choice_text_with_prefix.strip()

        answer_text = f"{answer_choice_text} as, the image is about {caption}"

        # Encode the image and text for the model
        # The processor typically handles image loading if given paths, but here we load with PIL first.
        encoding = self.processor(
            images=image, 
            text=question, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        # Tokenize the answer to create labels
        labels = self.processor.tokenizer.encode(
            answer_text, 
            max_length=50, # As used in the notebook
            padding="max_length", 
            truncation=True, 
            return_tensors='pt'
        )

        # Squeeze to remove the batch dimension added by the processor/tokenizer
        encoding["pixel_values"] = encoding["pixel_values"].squeeze(0)
        encoding["input_ids"] = encoding["input_ids"].squeeze(0)
        encoding["attention_mask"] = encoding["attention_mask"].squeeze(0)
        encoding["labels"] = labels.squeeze(0)

        return encoding
