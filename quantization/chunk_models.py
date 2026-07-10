import argparse
import os

import torch
from squeezellm.model_parse import (
    get_layers,
    get_named_modules,
    load_model,
    parse_model,
)
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_path", type=str, default=None, help="chunk the model and store"
)
parser.add_argument("--model", type=str, help="model to load")
parser.add_argument(
    "--model_type",
    type=str,
    default=None,
    help="model type",
    choices=["llama", "opt", "mistral", "qwen"],
)
parser.add_argument("--overwrite", action="store_true")


def is_loadable(path):
    if not os.path.exists(path):
        return False
    try:
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict) or not payload:
            return False
        del payload
        return True
    except Exception as exc:
        print(f"Existing chunk is not loadable and will be rewritten: {path} ({exc})", flush=True)
        return False


def atomic_save(payload, path):
    tmp_path = f"{path}.tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

args = parser.parse_args()
# if model type is not explicitly given, infer from the model name
model_type = args.model_type or parse_model(args.model)

# This path is only taken when we want to chunk the model and store it,
# which is used when '--output_path' is passed as an argument.
print(f"chunking the model: {args.model} and storing in {args.output_path}")
os.makedirs(args.output_path, exist_ok=True)
model = load_model(args.model, model_type)
layers = get_layers(model, model_type)
for i, layer in tqdm(enumerate(layers), total=len(layers)):
    output_file = os.path.join(args.output_path, f"layer_{i}.pt")
    if not args.overwrite and is_loadable(output_file):
        print(f"Skipping existing chunk: {output_file}", flush=True)
        continue
    data = {}
    for name, lin in get_named_modules(layer, model_type):
        data[name] = lin.weight.data
    atomic_save(data, output_file)
