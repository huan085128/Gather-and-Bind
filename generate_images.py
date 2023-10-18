import argparse
import os
import random
import yaml
import re

import torch
from pipeline_gather_and_bind import GatherAndBindPipeline
from ptp_utils import AttentionStore, view_images


def load_model(model_path):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    pipe = GatherAndBindPipeline.from_pretrained(model_path).to(device)

    return pipe


def generate(pipe, prompt, seed):
    controller = AttentionStore()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    generator = torch.Generator(device.type).manual_seed(seed)
    pipe.safety_checker = dummy
    outputs, indices_to_alter_all, entropy_results = pipe(prompt=prompt, height=512, width=512, 
               increment_token_indices=None,
               loss_weight=3.0,
               num_inference_steps=50,
               controller=controller,
               attention_res=16,
               generator=generator,  
               scale_factor=20,
               scale_range=(1, 0.5),
               if_refinement_step=True,
               thresholds={0: 5.4, 10: 5.2, 20: 4.8},
               max_iter_to_alter=25,
               loss_mode="jsd",
               if_show_attention_map=False,
               l2_no_grad=False,
               run_standard_sd=False)

    return outputs['images'][0]

def save_image(image, prompt, seed, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    file_name = f"{output_directory}/{prompt}_{seed}.png"
    image.save(file_name)

def extract_prompt_from_yaml(prompts_path, key):
    try:
        with open(prompts_path, 'r') as file:
            yaml_data = yaml.load(file, Loader=yaml.FullLoader)

        if key in yaml_data:
            data = yaml_data[key]
            sentence_pattern = r'[A-Za-z\s]+'

            def extract_sentences(data):
                prompts = []
                for prompt in data:
                    prompts.extend(re.findall(sentence_pattern, prompt))
                return prompts

            prompts = extract_sentences(data)
            return prompts
        else:
            return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None
    
def dummy(images, **kwargs): 
	return images, False

def generate_multi_images(model_path, prompts_path, concept, val_output_path):
    pipe = load_model(model_path)
    prompts = extract_prompt_from_yaml(prompts_path, concept)

    for prompt in prompts:
        print(prompt)
        save_dir = f'{val_output_path}/{concept}'
        save_path = os.path.join(save_dir, prompt.replace(" ", "_"))

        # Check if save_path directory exists
        if os.path.exists(save_path):
            image_count = len([f for f in os.listdir(save_path) if f.endswith('.png')])
            if image_count >= 64:
                print(f"Skipping prompt: {prompt} as it already has {image_count} images.")
                continue

            # Generate and save additional images to reach 64
            unique_seeds = random.sample(range(10000), 64 - image_count)
        else:
            os.makedirs(save_path)
            unique_seeds = random.sample(range(10000), 64)

        for seed in unique_seeds:
            image = generate(pipe, [prompt], seed)
            file_name = f"{save_path}/{seed}.png"
            image.save(file_name)
        
def generate_images(model_path, prompt, seed, val_output_path):
    pipe = load_model(model_path)
    save_path = os.path.join(val_output_path, prompt.replace(" ", "_"))
    os.makedirs(save_path, exist_ok=True)
    image = generate(pipe, [prompt], seed)
    file_name = f"{save_path}/{seed}.png"
    image.save(file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        default="a turtle and a yellow bowl"
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=2000
    )

    parser.add_argument(
        '--output_directory',
        type=str,
        default='./output'
    )

    parser.add_argument(
        '--model_path',
        type=str,
        default='/home/models/stable_diffusion_v1.5',
        help='The path to the model (this will download the model if the path doesn\'t exist)'
    )

    parser.add_argument(
        '--val_output_path',
        type=str,
        default='./val_output',
    )

    args = parser.parse_args()


    generate_images(args.model_path, args.prompt, args.seed, args.val_output_path)

