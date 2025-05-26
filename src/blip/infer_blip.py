import argparse
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
# from peft import PeftConfig, PeftModel # Potentially not needed if adapters load automatically

def main(args):
    """
    Main function to perform inference with a fine-tuned BLIP VQA model.
    """
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Processor
    try:
        print(f"Loading BLIP processor from Salesforce/blip-vqa-base...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    except Exception as e:
        print(f"Error loading BLIP processor: {e}")
        return

    # Load Model
    # If 'Model/blip-saved-model/' was saved using model.save_pretrained() on a PeftModel,
    # then BlipForQuestionAnswering.from_pretrained() should load the base model with adapters.
    try:
        print(f"Loading fine-tuned BLIP model from {args.model_path}...")
        model = BlipForQuestionAnswering.from_pretrained(args.model_path)
        model.to(device)
        model.eval() # Set model to evaluation mode
        print("Model loaded successfully and set to evaluation mode.")
    except Exception as e:
        print(f"Error loading fine-tuned model from {args.model_path}: {e}")
        print("Ensure the model_path contains a valid fine-tuned BLIP model with adapters,")
        print("typically saved using `model.save_pretrained()` on a PEFT-enabled model.")
        return

    # Load and process image
    try:
        print(f"Loading image from {args.image_path}...")
        raw_image = Image.open(args.image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at {args.image_path}")
        return
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Prepare inputs for the model
    question = args.question
    print(f"Question: {question}")

    try:
        # The processor expects a batch of images, so we pass the image in a list
        # However, for single image inference, some processors might handle it directly.
        # The notebook used `processor(image, question, ...)`
        # Let's ensure consistency with typical Hugging Face processor usage for single images
        inputs = processor(images=raw_image, text=question, return_tensors="pt").to(device)
        
        # If the model was trained with float16 (as in the fine-tuning notebook),
        # and if CUDA is available, cast inputs to float16.
        if device.type == 'cuda' and hasattr(model.config, 'torch_dtype') and model.config.torch_dtype == torch.float16:
            inputs = {k: v.to(torch.float16) if torch.is_floating_point(v) else v for k, v in inputs.items()}
        
    except Exception as e:
        print(f"Error processing image and question with processor: {e}")
        return

    # Perform inference
    print("Generating answer...")
    try:
        with torch.no_grad(): # Ensure no gradients are calculated during inference
            outputs = model.generate(**inputs)
    except Exception as e:
        print(f"Error during model generation: {e}")
        return
    
    # Decode and print the answer
    try:
        answer = processor.decode(outputs[0], skip_special_tokens=True)
        print(f"\nModel Answer: {answer}")
    except Exception as e:
        print(f"Error decoding output: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform inference with a fine-tuned BLIP VQA model.")
    
    parser.add_argument("--model_path", type=str, default="Model/blip-saved-model",
                        help="Path to the directory containing the fine-tuned BLIP model (adapters and config). Default: Model/blip-saved-model")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the input image file.")
    parser.add_argument("--question", type=str, required=True,
                        help="The question to ask about the image.")

    args = parser.parse_args()
    main(args)
