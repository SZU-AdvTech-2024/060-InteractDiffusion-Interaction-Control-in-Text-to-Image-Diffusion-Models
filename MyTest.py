from diffusers import DiffusionPipeline
import torch
import pickle

pipeline = DiffusionPipeline.from_pretrained(
    "interactdiffusion/diffusers-v1-2",
    trust_remote_code=True,
    variant="fp16", torch_dtype=torch.float16
)
pipeline = pipeline.to("cuda:5")


res_path = "DATA/hico_det_test.pkl"
res = pickle.load(open(res_path, "rb"))

sample_id=2
max_instance_num=1
count=0

sample=res[sample_id]
prompt=sample["prompt"]

instance_num=min(max_instance_num,len(sample["subject_phrases"]))

prompt_list=prompt.split(",")

text_prompt=",".join([prompt_list[i] for i in range(instance_num)])


subject_phrases = [sample["subject_phrases"][i] for i in range(instance_num)]
object_phrases = [sample["object_phrases"][i] for i in range(instance_num)]
action_phrases = [sample["action_phrases"][i] for i in range(instance_num)]
subject_boxes = [sample["subject_boxes"][i] for i in range(instance_num)]
object_boxes = [sample["object_boxes"][i] for i in range(instance_num)]


images = pipeline(
    prompt=text_prompt,
    interactdiffusion_subject_phrases=subject_phrases,
    interactdiffusion_object_phrases=object_phrases,
    interactdiffusion_action_phrases=action_phrases,
    interactdiffusion_subject_boxes=subject_boxes,
    interactdiffusion_object_boxes=object_boxes,
    interactdiffusion_scheduled_sampling_beta=1,
    output_type="pil",
    num_inference_steps=50,
    ).images

images[0].save(f'/data/xiaoliangqiu/project/InteractDiff/generation_samples/generation_hoi{sample_id}/output.png')