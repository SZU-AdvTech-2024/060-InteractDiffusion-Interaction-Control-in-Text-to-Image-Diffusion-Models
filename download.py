# Use a pipeline as a high-level helper
# from transformers import pipeline
#
# messages = [
#     {"role": "user", "content": "Who are you?"},
# ]
# pipe = pipeline("text-generation", model="meta-llama/Llama-3.1-8B-Instruct")
# pipe.save_pretrained("./Llama")
from diffusers import DiffusionPipeline
import torch


pipeline = DiffusionPipeline.from_pretrained(
    "interactdiffusion/diffusers-v1-2",
    trust_remote_code=True,
    variant="fp16", torch_dtype=torch.float16
)

pipeline.save_pretrained("./interact")