import sys
import os
import torch
import glob
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

def load_prompts(prompts_file_path="./prompts.txt"):
    """Load prompts from a file or return default."""
    default_prompt = "What is in this image?"
    if not os.path.exists(prompts_file_path):
        print(f"No prompts.txt found. Using default: '{default_prompt}'")
        return [default_prompt]

    with open(prompts_file_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]

    if not prompts:
        return [default_prompt]

    print(f"Loaded {len(prompts)} prompts from prompts.txt")
    return prompts

def create_output_directories(image_path, prompt_count):
    """Create output directories for each prompt."""
    base_dir = os.path.dirname(image_path.rstrip('/'))
    dir_name = os.path.basename(image_path.rstrip('/'))
    output_base = os.path.join(base_dir, f"{dir_name}_TXT")

    prompt_dirs = []
    for i in range(1, prompt_count + 1):
        prompt_dir = os.path.join(output_base, str(i))
        os.makedirs(prompt_dir, exist_ok=True)
        prompt_dirs.append(prompt_dir)

    return prompt_dirs

def test_model(model_path, image_path):
    """Test model on image(s) with prompts from prompts.txt and save outputs."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Using device: {device}")
    print(f"Loading model from {model_path}")

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        device_map="auto",  # Enables multi-GPU balancing
        torch_dtype=torch_dtype,
        attn_implementation="eager"
    )
    processor = AutoProcessor.from_pretrained(model_path)

    prompts = load_prompts()
    prompt_dirs = create_output_directories(image_path, len(prompts))

    if os.path.isdir(image_path):
        image_files = glob.glob(os.path.join(image_path, "*.jpg")) + \
                      glob.glob(os.path.join(image_path, "*.jpeg")) + \
                      glob.glob(os.path.join(image_path, "*.png"))
        print(f"Found {len(image_files)} images in {image_path}")
    else:
        image_files = [image_path]
        print(f"Processing single image: {image_path}")

    for img_file in image_files:
        filename = os.path.basename(img_file)
        base_filename = os.path.splitext(filename)[0]
        print(f"\nProcessing image: {img_file}")

        try:
            image = Image.open(img_file).convert("RGB")

            for prompt_idx, prompt_text in enumerate(prompts):
                prompt_dir = prompt_dirs[prompt_idx]
                output_file = os.path.join(prompt_dir, f"{base_filename}.txt")

                print(f"Using prompt {prompt_idx+1}/{len(prompts)}: '{prompt_text}'")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "source": image},
                            {"type": "text", "text": prompt_text}
                        ]
                    }
                ]

                prompt = processor.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                inputs = processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt"
                ).to(device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        do_sample=False,
                        num_beams=2,
                    )

                output_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                print(f"Output: {output_text}")

                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(output_text)

                print(f"Saved output to: {output_file}")

        except Exception as e:
            print(f"Error processing image {img_file}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python app.py /path/to/model/ /path/to/images/")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]
    test_model(model_path, image_path)
