import json
import os
import random
from pipeline_gather_and_bind import GatherAndBindPipeline
from ptp_utils import AttentionStore, view_images
import torch
import re
import yaml
import numpy as np


def update_json_with_results(result_dict, json_file_path):
    existing_data = {}

    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            existing_data = json.load(json_file)

    prompt = result_dict['prompt']
    seed = result_dict['seed']

    if prompt not in existing_data:
        existing_data[prompt] = {}

    if seed not in existing_data[prompt]:
        existing_data[prompt][seed] = {}

    for key, value in result_dict.items():
        if key not in ['prompt', 'seed']:
            existing_data[prompt][seed][key] = value

    with open(json_file_path, 'w') as json_file:
        json.dump(existing_data, json_file, indent=4)

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
    
def extract_prompts_and_seeds_from_yaml(yaml_path):
    try:
        with open(yaml_path, 'r') as file:
            yaml_data = yaml.load(file, Loader=yaml.FullLoader)

        prompts_with_seeds = {} 

        for category, prompts in yaml_data.items():
            if category == "seeds":
                continue

            prompts_with_seeds[category] = {}  
            for prompt in prompts:
                seeds = yaml_data["seeds"].get(category, {}).get(prompt, [])
                prompts_with_seeds[category][prompt] = seeds

        return prompts_with_seeds
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None
    
def load_model(model_path):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    pipe = GatherAndBindPipeline.from_pretrained(model_path).to(device)

    return pipe

def generate(pipe, prompt, seed, run_standard_sd):
    controller = AttentionStore()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    generator = torch.Generator(device.type).manual_seed(seed)
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
               max_iter_to_alter=50,
               loss_mode="jsd",
               if_show_attention_map=False,
               l2_no_grad=False,
               run_standard_sd=run_standard_sd)

    return outputs['images'][0], entropy_results

def generate_multi(model_path, prompts_path: str, key: str, json_file_path: str,
                   val_output_path: str, run_standard_sd: bool = False):
    
    pipe = load_model(model_path)
    prompts_and_seeds = extract_prompts_and_seeds_from_yaml(yaml_path=prompts_path)
    for category, prompts in prompts_and_seeds.items():
        if category == key:
            for prompt, seeds in prompts.items():
                for seed in seeds:
                    image, entropy_results = generate(pipe, [prompt], seed, run_standard_sd=run_standard_sd)
                    save_path = os.path.join(f'{val_output_path}/{key}', prompt.replace(" ", "_"))
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    file_name = f"{save_path}/{seed}.png"
                    image.save(file_name)
                    entropy_results['prompt'] = prompt
                    entropy_results['seed'] = seed
                    update_json_with_results(entropy_results, json_file_path)


if __name__ == '__main__':

    model_path = "/ai/workshop/stable-diffusion-v1-5"   
    json_file_path = 'animals_results.json'
    val_output_path = 'calculate_entropies_output'
    prompts_path = '/ai/workshop/pipeline/random_seeds.yaml'
    key = 'animals'
    run_standard_sd = True

    generate_multi(model_path=model_path, prompts_path=prompts_path, key=key, json_file_path=json_file_path,
                   val_output_path=val_output_path, run_standard_sd=run_standard_sd)


