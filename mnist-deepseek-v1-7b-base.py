# https://huggingface.co/deepseek-ai/deepseek-vl-7b-base
# https://x.com/i/grok?conversation=1896539730960880091
import os
import torch
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
from torchvision import datasets
from PIL import Image
import numpy as np
from datetime import datetime
import argparse

# Path configurations
BASE_PATH = "/users/aczd097"
MNIST_PATH = os.path.join(BASE_PATH, "mnist")
RESULTS_PATH = os.path.join(MNIST_PATH, "results")
RAW_PATH = os.path.join(MNIST_PATH, "raw")
TMP_IMAGE_DIR = "/users/aczd097/git/DeepSeek-VL/tmp_images"
TMP_IMAGE_PATH = os.path.join(TMP_IMAGE_DIR, "tmp_img.png")

# MNIST class mapping
MNIST_CLASSES = {str(i): i for i in range(10)}
MNIST_IDX_TO_CLASS = {v: k for k, v in MNIST_CLASSES.items()}

# Model setup
model_path = "deepseek-ai/deepseek-vl-7b-base"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer
vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

def setup_argparse():
    parser = argparse.ArgumentParser(description='MNIST Classification with DeepSeek VL')
    parser.add_argument('--resume-from', type=str, help='Path to saved results file to resume from')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    return parser.parse_args()

def load_mnist_datasets():
    for path in [MNIST_PATH, RESULTS_PATH, RAW_PATH, TMP_IMAGE_DIR]:
        os.makedirs(path, exist_ok=True)
    print("Loading MNIST datasets...")
    train_dataset = datasets.MNIST(RAW_PATH, train=True, download=True)
    test_dataset = datasets.MNIST(RAW_PATH, train=False, download=True)
    return train_dataset, test_dataset

def extract_class(response, debug=False):
    try:
        response = response.strip().lower()
        if debug:
            print(f"Full model response: '{response}'")
        for digit in MNIST_CLASSES:
            if digit in response:
                if debug:
                    print(f"Parsed digit: '{digit}' (class {MNIST_CLASSES[digit]})")
                return MNIST_CLASSES[digit]
        if debug:
            print("No valid digit found, returning unknown (10)")
        return 10
    except Exception as e:
        if debug:
            print(f"Error parsing response: {str(e)}")
        return 10

def process_image(image, processor, model, debug=False):
    """Process a single MNIST image by saving to disk temporarily."""
    # Convert to PIL if needed and save to temporary file
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image.save(TMP_IMAGE_PATH)
    
    # Prepare conversation with file path
    class_options = ", ".join(MNIST_CLASSES.keys())
    conversation = [
        {
            "role": "User",
            "content": f"<image_placeholder>What digit is shown in this image? Choose one: {class_options}. Respond with only the digit.",
            "images": [TMP_IMAGE_PATH]
        },
        {"role": "Assistant", "content": ""}
    ]
    
    # Load and process inputs
    pil_images = load_pil_images(conversation)
    prepare_inputs = processor(conversations=conversation, images=pil_images, force_batchify=True).to(model.device)
    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
    
    # Generate response
    outputs = model.language_model.generate(
        inputs_embeds=inputs_embeds, attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=5, do_sample=False, use_cache=True
    )
    
    # Decode response
    response = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    
    # Clean up temporary file
    if os.path.exists(TMP_IMAGE_PATH):
        os.remove(TMP_IMAGE_PATH)
    
    # Extract class with debug info
    return extract_class(response, debug=debug)

def load_previous_results(filepath):
    try:
        data = np.load(filepath, allow_pickle=True).item()
        indices = data['indices']
        true_labels = data['true_labels']
        predicted_labels = data['predicted_labels']
        timestamp = data.get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
        results = {'true_labels': true_labels, 'predicted_labels': predicted_labels}
        last_idx = max(indices) if indices else -1
        next_idx = last_idx + 1
        print(f"Loaded {len(indices)} processed images. Resuming from image {next_idx}")
        return results, next_idx, timestamp
    except Exception as e:
        print(f"Error loading previous results: {str(e)}")
        return {'true_labels': [], 'predicted_labels': []}, 0, datetime.now().strftime("%Y%m%d_%H%M%S")

def save_results(results_dict, save_path, set_type, timestamp):
    indices = sorted(list(range(len(results_dict['true_labels']))))
    data = {
        'indices': indices,
        'true_labels': results_dict['true_labels'],
        'predicted_labels': results_dict['predicted_labels'],
        'timestamp': timestamp
    }
    save_file = os.path.join(save_path, f'mnist_{set_type}_deepseek-vl-7b_{timestamp}.npy')
    np.save(save_file, data)

def main():
    args = setup_argparse()
    train_dataset, test_dataset = load_mnist_datasets()
    
    # Process test set only for simplicity
    dataset = test_dataset
    set_type = 'testing'
    
    results, start_idx, timestamp = (None, 0, datetime.now().strftime("%Y%m%d_%H%M%S"))
    if args.resume_from:
        results, start_idx, timestamp = load_previous_results(args.resume_from)
    
    if results is None:
        results = {'true_labels': [], 'predicted_labels': []}
    
    total_images = len(dataset)
    print(f"Processing {set_type} dataset ({total_images} images) from index {start_idx}...")
    
    for idx in range(start_idx, total_images):
        if idx % 100 == 0 and not args.debug:
            print(f"Processing image {idx}/{total_images} ({idx/total_images*100:.1f}%)")
        
        image, label = dataset[idx]
        predicted_class = process_image(image, vl_chat_processor, vl_gpt, debug=args.debug)
        
        if args.debug:
            print(f"\n{'='*50}")
            print(f"Image {idx}")
            print(f"True label: {label} ({MNIST_IDX_TO_CLASS[label]})")
            print(f"Predicted class: {predicted_class} ({MNIST_IDX_TO_CLASS.get(predicted_class, 'unknown')})")
            print(f"{'='*50}\n")
            input("Press Enter to continue...")
        
        results['true_labels'].append(label)
        results['predicted_labels'].append(predicted_class)
        
        if (idx - start_idx + 1) % 1000 == 0 or idx == total_images - 1:
            save_results(results, RESULTS_PATH, set_type, timestamp)
    
    save_results(results, RESULTS_PATH, set_type, timestamp)
    accuracy = sum(np.array(results['true_labels']) == np.array(results['predicted_labels'])) / len(results['true_labels'])
    print(f"{set_type.capitalize()} accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
