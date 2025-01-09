from transformers import CLIPTextModel,CLIPModel,CLIPProcessor
import torch
import os

pipe = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
prepare_batch_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
pipe.save_pretrained("clip_inference/")
prepare_batch_processor.save_pretrained("clip_inference/")