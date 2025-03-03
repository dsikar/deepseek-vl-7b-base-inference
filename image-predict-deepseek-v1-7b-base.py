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
CIFAR10_PATH = os.path.join(BASE_PATH, "cifar10")
RESULTS_PATH_MNIST = os.path.join(MNIST_PATH, "results")
RESULTS_PATH_CIFAR10 = os.path.join(CIFAR10_PATH, "results")
RAW_PATH_MNIST = os.path.join(MNIST_PATH, "raw")
RAW_PATH_CIFAR10 = os.path.join(CIFAR10_PATH, "raw")
TMP_IMAGE_DIR = "/users/aczd097/git/DeepSeek-VL/tmp_images"
TMP_IMAGE_PATH = os.path.join(TMP_IMAGE_DIR, "tmp_img.png")

# Class mappings
MNIST_CLASSES = {str(i): i for i in range(10)}
MNIST_IDX_TO_CLASS = {v: k for k, v in MNIST_CLASSES.items()}
CIFAR10_CLASSES = {
    'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
    'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
}
CIFAR10_IDX_TO_CLASS = {v: k for k, v in CIFAR10_CLASSES.items()}

# Model setup
model_path = "deepseek-ai/deepseek-vl-7b-base"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer
vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl"),g_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

def setup_argparse():
    parser = argparse.ArgumentParser(description='MNIST/CIFAR-10 Classification with DeepSeek VL')
    parser.add_argument('--resume-from', type=str, help='Path to saved results file to resume from')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--cifar10', action='store_true', help='Process CIFAR-10 instead of MNIST (default)')
    return parser.parse_args()

def load_datasets(use_cifar10=False):
    """Load either MNIST or CIFAR-10 datasets based on flag."""
    if use_cifar10:
        dataset_path = CIFAR10_PATH
        raw_path = RAW_PATH_CIFAR10
        results_path = RESULTS_PATH_CIFAR10
        print("Loading CIFAR-10 datasets...")
        train_dataset = datasets.CIFAR10(raw_path, train=True, download=True)
        test_dataset = datasets.CIFAR10(raw_path, train=False, download=True)
        class_dict = CIFAR10_CLASSES
        idx_to_class = CIFAR10_IDX_TO_CLASS
    else:
        dataset_path = MNIST_PATH
        raw_path = RAW_PATH_MNIST
        results_path = RESULTS_PATH_MNIST
        print("Loading MNIST datasets...")
        train_dataset = datasets.MNIST(raw_path, train=True, download=True)
        test_dataset = datasets.MNIST(raw_path, train=False, download=True)
        class_dict = MNIST_CLASSES
        idx_to_class = MNIST_IDX_TO_CLASS
    
    for path in [dataset_path, results_path, raw_path, TMP_IMAGE_DIR]:
        os.makedirs(path, exist_ok=True)
    
    return train_dataset, test_dataset, class_dict, idx_to_class, results_path

def extract_class(response, class_dict, debug=False):
    """Extract class from model response based on dataset."""
    try:
        response = response.strip().lower()
        if debug:
            print(f"Full model response: '{response}'")
        for class_name in class_dict:
            if class_name in response:
                if debug:
                    print(f"Parsed class: '{class_name}' (index {class_dict[class_name]})")
                return class_dict[class_name]
        if debug:
            print("No valid class found, returning unknown (10)")
        return 10
    except Exception as e:
        if debug:
            print(f"Error parsing response: {str(e)}")
        return 10

def process_image(image, processor, model, class_dict, debug=False):
    """Process a single image by saving to disk temporarily."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image.save(TMP_IMAGE_PATH)
    
    class_options = ", ".join(class_dict.keys())
    conversation = [
        {
            "role": "User",
            "content": f"<image_placeholder>What is shown in this image? Choose one: {class_options}. Respond with only the class name.",
            "images": [TMP_IMAGE_PATH]
        },
        {"role": "Assistant", "content": ""}
    ]
    
    pil_images = load_pil_images(conversation)
    prepare_inputs = processor(conversations=conversation, images=pil_images, force_batchify=True).to(model.device)
    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
    
    outputs = model.language_model.generate(
        inputs_embeds=inputs_embeds, attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=20, do_sample=False, use_cache=True  # Increased for CIFAR-10 longer names
    )
    
    response = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    
    if os.path.exists(TMP_IMAGE_PATH):
        os.remove(TMP_IMAGE_PATH)
    
    return extract_class(response, class_dict, debug=debug)

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

def save_results(results_dict, save_path, set_type, timestamp, use_cifar10=False):
    indices = sorted(list(range(len(results_dict['true_labels']))))
    data = {
        'indices': indices,
        'true_labels': results_dict['true_labels'],
        'predicted_labels': results_dict['predicted_labels'],
        'timestamp': timestamp
    }
    dataset_name = 'cifar10' if use_cifar10 else 'mnist'
    save_file = os.path.join(save_path, f'{dataset_name}_{set_type}_deepseek-vl-7b_{timestamp}.npy')
    np.save(save_file, data)

def main():
    args = setup_argparse()
    train_dataset, test_dataset, class_dict, idx_to_class, results_path = load_datasets(args.cifar10)
    
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
        predicted_class = process_image(image, vl_chat_processor, vl_gpt, class_dict, debug=args.debug)
        
        if args.debug:
            print(f"\n{'='*50}")
            print(f"Image {idx}")
            print(f"True label: {label} ({idx_to_class[label]})")
            print(f"Predicted class: {predicted_class} ({idx_to_class.get(predicted_class, 'unknown')})")
            print(f"{'='*50}\n")
            input("Press Enter to continue...")
        
        results['true_labels'].append(label)
        results['predicted_labels'].append(predicted_class)
        
        if (idx - start_idx + 1) % 1000 == 0 or idx == total_images - 1:
            save_results(results, results_path, set_type, timestamp, use_cifar10=args.cifar10)
    
    save_results(results, results_path, set_type, timestamp, use_cifar10=args.cifar10)
    accuracy = sum(np.array(results['true_labels']) == np.array(results['predicted_labels'])) / len(results['true_labels'])
    print(f"{set_type.capitalize()} accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
