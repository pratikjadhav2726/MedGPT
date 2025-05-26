import argparse
import torch
from PIL import Image
import os # For os.path.join

from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    process_images, 
    tokenizer_image_token,
    get_model_name_from_path
)
from llava.constants import (
    IMAGE_TOKEN_INDEX
)
from llava.utils import disable_torch_init
from llava.conversation import conv_templates, SeparatorStyle


from peft import PeftModel 

# Import create_prompt from llava_dataset.py
from llava_dataset import create_prompt

def main(args):
    """
    Main function to perform inference with a LLaVA model (base or fine-tuned with LoRA).
    """
    disable_torch_init() 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine the base model ID and if we are loading adapters
    load_adapters = args.adapter_path is not None
    base_model_id = args.base_model_id if load_adapters else args.model_path_or_id
    
    try:
        print(f"Loading base model components from: {base_model_id}...")
        model_name_for_loading = get_model_name_from_path(base_model_id) 
        
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=base_model_id,
            model_base=None, 
            model_name=model_name_for_loading,
            load_4bit=args.load_in_4bit,
            device_map='auto' 
        )
        print("Base model components loaded.")
        
    except Exception as e:
        print(f"Error loading base model from {base_model_id}: {e}")
        return

    if load_adapters:
        try:
            print(f"Loading LoRA adapters from: {args.adapter_path}...")
            model = PeftModel.from_pretrained(model, args.adapter_path)
            # Ensure model is on the correct device after adapter loading if not using device_map='auto' for PeftModel
            # For LLaVA with 4-bit, the base model is already on device(s) via device_map.
            # PeftModel typically respects this.
            print(f"LoRA adapters from {args.adapter_path} loaded and applied.")
        except Exception as e:
            print(f"Error loading LoRA adapters from {args.adapter_path}: {e}")
            print("Ensure the adapter_path is correct and contains valid LoRA adapter files.")
            return
            
    model.eval() 
    print("Model set to evaluation mode.")

    # Load and process image
    try:
        print(f"Loading image from {args.image_file}...")
        image = Image.open(args.image_file).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at {args.image_file}")
        return
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    image_dtype = torch.float16 
    if hasattr(model.config, 'torch_dtype'):
        if model.config.torch_dtype == torch.bfloat16:
            image_dtype = torch.bfloat16
        elif model.config.torch_dtype == torch.float32: 
            image_dtype = torch.float32
            
    image_tensor = process_images([image], image_processor, model.config).to(
        model.device, dtype=image_dtype
    )
    image_sizes = [image.size]

    # Create prompt
    # The model name for conversation template should ideally be derived from the base model's config
    # or the model_name_for_loading used above.
    # If model.config.mm_vision_tower (or similar) indicates the vision tower,
    # and model.config.model_type indicates the LLM (e.g. 'mistral'), we can infer conv_mode.
    # get_model_name_from_path(base_model_id) is a good heuristic.
    prompt_model_name = get_model_name_from_path(base_model_id)
    full_prompt = create_prompt(args.prompt, model.config, model_name=prompt_model_name)
    
    input_ids = (
        tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0) 
        .to(model.device)
    )

    print("Generating answer...")
    try:
        with torch.inference_mode(): 
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes, 
                do_sample=args.temperature > 0, 
                temperature=args.temperature if args.temperature > 0 else None, 
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
    except Exception as e:
        print(f"Error during model generation: {e}")
        return
    
    try:
        outputs_decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        # Attempt to extract only the assistant's response
        # This logic needs to be robust to the conversation template
        conv_mode = "mistral_instruct" # Default, adjust if needed based on prompt_model_name
        if "vicuna" in prompt_model_name.lower():
            conv_mode = "vicuna_v1"
        elif not ("mistral" in prompt_model_name.lower() or "llava-v1.6" in prompt_model_name.lower()):
            conv_mode = "llava_v1"

        conv = conv_templates[conv_mode].copy()
        sep = conv.sep if conv.sep_style == SeparatorStyle.TWO else conv.sep2
        
        # The output includes the prompt. Find the start of the assistant's actual reply.
        parts = outputs_decoded.split(sep)
        assistant_response = parts[-1].strip() # Usually the last part after the final separator
        
        # Further refinement: if the prompt's user turn is in the output, remove it.
        # This is a bit heuristic as the model might not perfectly repeat the prompt.
        # A simple way: check if the assistant's response starts with the user prompt.
        # For now, the above split is a common approach.

        print(f"\nModel Answer:\n{assistant_response}")

    except Exception as e:
        print(f"Error decoding or processing output: {e}")
        print(f"Raw decoded output: {outputs_decoded if 'outputs_decoded' in locals() else 'N/A'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform inference with a LLaVA model.")
    
    parser.add_argument("--model_path_or_id", type=str, required=True,
                        help="Hugging Face ID of a base LLaVA model (if no adapters) OR path to a directory containing a full LLaVA model (if no adapters). If --adapter_path is specified, this argument is IGNORED and --base_model_id is used for the base.")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="(Optional) Path to fine-tuned LoRA adapter directory. If specified, --base_model_id will be used to load the base model.")
    parser.add_argument("--base_model_id", type=str, default="liuhaotian/llava-v1.6-mistral-7b",
                        help="Base model ID to use when --adapter_path is specified. Default: liuhaotian/llava-v1.6-mistral-7b")
    
    parser.add_argument("--image_file", type=str, required=True,
                        help="Path to the input image file.")
    parser.add_argument("--prompt", type=str, required=True,
                        help="The prompt or question to ask about the image.")
    
    parser.add_argument("--load_in_4bit", action='store_true', default=True,
                        help="Load the base model in 4-bit precision (default).")
    parser.add_argument("--no_load_in_4bit", action='store_false', dest='load_in_4bit',
                        help="Do not load the base model in 4-bit precision.")

    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Temperature for sampling. Set to 0 for greedy decoding. Default: 0.2")
    parser.add_argument("--num_beams", type=int, default=1,
                        help="Number of beams for beam search. 1 means no beam search. Default: 1")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of new tokens to generate. Default: 512")

    args = parser.parse_args()
    main(args)
