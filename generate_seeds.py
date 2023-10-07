import random
import yaml

yaml_file_path = '/ai/workshop/pipeline/a.e_prompts.yaml'

with open(yaml_file_path, 'r') as yaml_file:
    original_yaml_data = yaml.safe_load(yaml_file)

num_seeds = 64
seeds_per_prompt = {}

for category in ["animals"]:
    category_seeds = {}
    for prompt in original_yaml_data[category]:
        seed_list = [random.randint(0, 10000) for _ in range(num_seeds)]
        category_seeds[prompt] = seed_list

    seeds_per_prompt[category] = category_seeds

original_yaml_data["seeds"] = seeds_per_prompt

output_yaml_file = "random_seeds.yaml"

with open(output_yaml_file, "w") as yaml_out:
    yaml.dump(original_yaml_data, yaml_out, default_flow_style=False)

print(f"Saved to {output_yaml_file}")
