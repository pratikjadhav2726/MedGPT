import os
import re
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from llava.mm_utils import (
    tokenizer_image_token,
    process_images, # Used by process_prepare_img
)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_PLACEHOLDER
)
from llava.conversation import conv_templates

def create_prompt(query: str, model_config, caption: str = None, model_name: str = "llava-v1.6-mistral-7b"):
    """
    Creates a LLaVA-formatted prompt using conversation templates.

    Args:
        query (str): The user's query or instruction.
        model_config: The model's configuration object (model.config), used to check mm_use_im_start_end.
        caption (str, optional): The target caption or model's response part. Defaults to None.
        model_name (str, optional): The model name to determine conversation template.
                                     Defaults to "llava-v1.6-mistral-7b", implies "mistral_instruct".

    Returns:
        str: The fully formatted prompt string.
    """
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    
    # Prepend image token if not already in query
    if IMAGE_PLACEHOLDER in query:
        if model_config.mm_use_im_start_end:
            query = re.sub(IMAGE_PLACEHOLDER, image_token_se, query)
        else:
            query = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, query)
    else:
        # Default behavior: image token first, then newline, then query
        if model_config.mm_use_im_start_end:
            query = image_token_se + "\n" + query
        else:
            query = DEFAULT_IMAGE_TOKEN + "\n" + query

    # Determine conversation mode based on model name (simplified)
    if "mistral" in model_name.lower() or "llava-v1.6" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "vicuna" in model_name.lower():
        conv_mode = "vicuna_v1"
    else:
        # Fallback or more sophisticated mode detection might be needed
        conv_mode = "llava_v1" 
        print(f"Warning: Using default conv_mode '{conv_mode}' for model {model_name}")


    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], query) # User's turn
    conv.append_message(conv.roles[1], caption) # Assistant's turn (caption or None for generation)
    
    return conv.get_prompt()

def process_prepare_img(image_filenames: list, image_dir: str, image_processor, model_config, device):
    """
    Loads, processes, and prepares a list of images.

    Args:
        image_filenames (list of str): List of image filenames (e.g., ['img1.jpg', 'img2.png']).
        image_dir (str): Directory where images are stored.
        image_processor: The LLaVA image processor.
        model_config: The model's configuration object (model.config).
        device: The torch device to send the tensor to.

    Returns:
        tuple: (images_tensor, image_sizes)
               - images_tensor: Processed images as a PyTorch tensor.
               - image_sizes: List of original (width, height) of the images.
    """
    loaded_images = []
    for filename in image_filenames:
        full_path = os.path.join(image_dir, filename)
        try:
            image = Image.open(full_path).convert('RGB')
            loaded_images.append(image)
        except FileNotFoundError:
            print(f"Error: Image file not found at {full_path}")
            # Handle missing images as needed, e.g., skip or use a placeholder
            raise
        except Exception as e:
            print(f"Error loading image {full_path}: {e}")
            raise
            
    if not loaded_images:
        # Return empty tensors or handle as appropriate if no images could be loaded
        return torch.tensor([]).to(device), []

    images_tensor = process_images(
        loaded_images, image_processor, model_config
    ).to(
        device, dtype=torch.bfloat16 if model_config.torch_dtype == torch.bfloat16 else torch.float16
    )
    image_sizes = [img.size for img in loaded_images]
    return images_tensor, image_sizes


def tokenize_and_create_labels_for_llava(batch, tokenizer, image_processor, model_config, image_dir: str, ignore_index: int = -100, device: str = "cuda"):
    """
    Collate function to tokenize a batch of examples (image + caption) and create labels for LLaVA fine-tuning.
    The 'image' key in the batch items should be the filename of the image.
    The 'caption' key in the batch items should be the target text.
    """
    pad_token_id = tokenizer.pad_token_id
    
    # Extract image filenames and captions from the batch
    image_filenames_in_batch = [item['image'] for item in batch]
    captions_in_batch = [item['caption'] for item in batch]

    # Process images
    # model_config is passed to determine dtype and other processing details
    image_tensor, image_sizes = process_prepare_img(
        image_filenames_in_batch, image_dir, image_processor, model_config, device
    )

    # Fixed query for captioning-style fine-tuning
    query = "Describe the image in detail." 
    # Alternative: "What is this picture about?" or pass query as part of the batch item

    # Create prompts and tokenize
    prompts_without_caption = [create_prompt(query, model_config, None) for _ in captions_in_batch]
    tokenized_prompts_without_caption = [
        tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').to(device) 
        for p in prompts_without_caption
    ]

    prompts_with_caption = [create_prompt(query, model_config, cap) for cap in captions_in_batch]
    tokenized_prompts_with_caption = [
        tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').to(device)
        for p in prompts_with_caption
    ]

    # Pad the input_ids (which include the target caption)
    input_ids_list = [tcwc.squeeze(0) for tcwc in tokenized_prompts_with_caption]
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id).to(device)
    
    attention_mask = (input_ids != pad_token_id).long().to(device)
    
    labels = torch.full_like(input_ids, fill_value=ignore_index)
    
    for i, tokenized_prompt_only in enumerate(tokenized_prompts_without_caption):
        len_prompt_only = tokenized_prompt_only.shape[1]
        # The labels start after the prompt part (image + query)
        # and are the tokens of the caption part from the full sequence (input_ids[i])
        labels[i, len_prompt_only:] = input_ids[i, len_prompt_only:]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "images": image_tensor, # This key name 'images' is expected by LLaVA models
        "image_sizes": image_sizes, # May or may not be used by the model during training
        "labels": labels
    }

if __name__ == '__main__':
    # Example usage (requires a dummy model_config, tokenizer, image_processor, etc.)
    print("llava_dataset.py: Contains utility functions for LLaVA data processing and prompt creation.")
    
    # This block would require more setup to be runnable (tokenizer, model_config etc.)
    # print("\nTesting create_prompt (requires model.config):")
    # class DummyConfig:
    #     mm_use_im_start_end = True
    #     torch_dtype = torch.float32 # or bfloat16
    # dummy_config = DummyConfig()
    # test_query = "<image>\nWhat is this?"
    # prompt = create_prompt(test_query, dummy_config, "This is a test caption.")
    # print(f"Generated prompt: {prompt}")

    # print("\nNote: process_prepare_img and tokenize_and_create_labels_for_llava require more complex setup for testing here.")
    pass
