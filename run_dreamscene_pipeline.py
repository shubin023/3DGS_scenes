"""
End-to-end pipeline:
prompt  ->  Qwen3-14B  ->  YAML  ->  docker cp  ->  DreamScene run  ->  .ply out
"""

import subprocess, pathlib, datetime as dt, textwrap, yaml, argparse, os, sys
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

p = argparse.ArgumentParser()
p.add_argument("prompt", help="Natural-language scene description")
p.add_argument("--container", default="d80f9c8b8be4", help="Running DreamScene container ID / name")
args = p.parse_args()

MODEL_ID = "Qwen/Qwen3-14B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model     = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto",
                                                 torch_dtype="auto", trust_remote_code=True)

prompt = textwrap.dedent("""
SYSTEM
You are DreamScene-GPT, an assistant that produces **valid YAML ONLY** for the DreamScene 3-D scene-generation project.  
Never wrap the YAML in markdown fences. Never add commentary.  
Follow the template exactly, replacing every ‹PLACEHOLDER› with concrete values.

USER
### Instructions
1. Copy the entire TEMPLATE below.
2. Replace:
   • ‹EXP_NAME›               – short slug for the run
   • ‹SCENE_PROMPT›           – the full user prompt (positive)
   • ‹NEGATIVE_PROMPT›        – optional negative prompt or leave ""
   • ‹SCENE_NAME›             – file-safe scene name
   • ‹CAM_POSE_METHOD›        – "indoor" or "outdoor"
   • For every distinct 3-D object in the prompt:
       id              – snake_case noun
       text            – DSLR-style prompt
       init_prompt     – concise noun phrase
       center          – [x,y,z] in metres (fit in a 5×5×3 m room for indoor, 30×30×4 m for outdoor)
       rotation        – [roll,pitch,yaw] deg, default [0,0,0]
       scale           – [sx,sy,sz], default [1,1,1]
3. Leave all optimisation and camera hyper-params unchanged.
4. Output YAML **only**—no extra keys, no markdown.

### TEMPLATE  (fill in the placeholders)
# CUDA_VISIBLE_DEVICES=6,7 python main.py --config configs/scenes/sample_outdoor.yaml
optimizationParams:
  iterations: 1500
  position_lr_init: 0.0008
  position_lr_final: 0.000016
  feature_lr: 0.020
  feature_lr_final: 0.0030
  scaling_lr: 0.005
  scaling_lr_final: 0.001
  rotation_lr: 0.001
  rotation_lr_final: 0.0002

  densify_grad_threshold: 0.00075

  densify_from_iter: 100
  densify_until_iter: 1400
  densification_interval: 100
  opacity_reset_interval: 600
  style_prompt: ''
  style_negative_prompt: ""

reconOptimizationParams:
  iterations: 40
  position_lr_init: 0.0008
  position_lr_final: 0.000008
  feature_lr: 0.005
  feature_lr_final: 0.004
  opacity_lr: 0.05
  scaling_lr: 0.005
  scaling_lr_final: 0.002
  rotation_lr: 0.001
  rotation_lr_final: 0.0002

  percent_dense: 0.01
  densify_grad_threshold: 0.001

  densify_from_iter: 0
  densify_until_iter: 50
  opacity_reset_interval: 3000
  as_latent_ratio: 0.0

generateCamParams:
  phi_range: [-180, 180]
  max_phi_range: [-180, 180]
  rand_cam_gamma: 1.

  theta_range: [45, 105]
  max_theta_range: [45, 105]

  radius_range: [5.2, 5.5] 
  max_radius_range: [3.5, 5.0]
  default_radius: 3.5

  default_fovy: 0.55
  fovy_range: [0.32, 0.60]
  max_fovy_range: [0.16, 0.60]

sceneOptimizationParams:
  iterations: 1000
  position_lr_init: 0.0008
  position_lr_final: 0.000025
  feature_lr: 0.020
  feature_lr_final: 0.005
  scaling_lr: 0.005
  scaling_lr_final: 0.005
  rotation_lr: 0.001
  rotation_lr_final: 0.0004
  densify_grad_threshold: 0.001

  densify_from_iter: 100
  densify_until_iter: 1400
  densification_interval: 100
  opacity_reset_interval: 30000
  max_point_number: 3000000
  lambda_scale: 0.5
  lambda_tv: 1.0
  lambda_tv_depth: 0.0
  style_prompt: ""
  style_negative_prompt: ""

reconSceneOptimizationParams:
  iterations: 1000
  position_lr_init: 0.0008
  position_lr_final: 0.000008
  feature_lr: 0.005
  feature_lr_final: 0.004
  opacity_lr: 0.05
  scaling_lr: 0.005 
  scaling_lr_final: 0.002
  rotation_lr: 0.001
  rotation_lr_final: 0.0002

  percent_dense: 0.01
  densify_grad_threshold: 0.001

  densify_from_iter: 0
  densify_until_iter: 50
  opacity_reset_interval: 300000
  as_latent_ratio: 0.0
  max_point_number: 3000000

fineSceneOptimizationParams:
  iterations: 2000
  position_lr_init: 0.00008
  position_lr_final: 0.000008
  feature_lr: 0.001
  feature_lr_final: 0.0005
  scaling_lr: 0.001
  scaling_lr_final: 0.0005
  rotation_lr: 0.0002
  rotation_lr_final: 0.00005
  densify_grad_threshold: 0.001

  densify_from_iter: 0
  densify_until_iter: 900
  densification_interval: 100
  opacity_reset_interval: 30000
  max_point_number: 3000000
  lambda_scale: 0.5
  lambda_tv: 1.0
  lambda_tv_depth: 0.0
  style_prompt: ''
  style_negative_prompt: ""

sceneGenerateCamParams:
  phi_range: [-180, 180]
  max_phi_range: [-180, 180]
  rand_cam_gamma: 1.

  theta_range: [45, 105]
  max_theta_range: [45, 105]

  radius_range: [5.2, 5.5] 
  max_radius_range: [3.5, 5.0]
  default_radius: 4.5

  default_fovy: 0.96
  fovy_range: [0.70, 1.20]
  max_fovy_range: [0.70, 1.00]

guidanceParams:
  model_key: 'stabilityai/stable-diffusion-2-1-base'
  perpneg: false
  C_batch_size: 4
  lambda_guidance: 0.1
  noise_seed: 0

  delta_t: 50
  guidance: 'MTSD'
  g_device: 'cuda:1'

  random_delta: True
  vis_interval: 100

visualize_samples: True

modelParams:
  bg_aug_ratio: 0.66

seed: 0

log:
  exp_name: ‹EXP_NAME›
  eval_only: False

scene_configs:
  objects:
    # Repeat this block for every object discovered in the prompt
    - id: ‹OBJ_ID›
      sh_degree: 1
      text: ‹OBJ_TEXT›
      negative_text: ''
      image: ''
      init_guided: 'pointe'
      init_prompt: ‹OBJ_INIT_PROMPT›
      cam_pose_method: 'object'
      use_pointe_rgb: false
      num_pts: 20000
      radius: 0.5

  scene:
    scene_name: ‹SCENE_NAME›
    sh_degree: 1
    cam_pose_method: ‹CAM_POSE_METHOD›
    scene_text: ‹SCENE_PROMPT›
    negative_text: ‹NEGATIVE_PROMPT›
    zero_ground: true
    floor_init_color: [64,222,90]
    env_init_color: [200,160,160]
    radius: [15,15,4]

    scene_composition:
      - id: ‹OBJ_ID›
        params:
          - center: ‹OBJ_CENTER›
            rotation: ‹OBJ_ROTATION›
            scale: ‹OBJ_SCALE›

mode_args:
  prune_decay: 0.8
  v_pow: 0.1
  prune_percent: 0.5

### SCENE_PROMPT
{user_prompt}
""").format(user_prompt=args.prompt)

messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)

yaml.safe_load(content)

ts   = dt.datetime.now().strftime("%Y%m%d_%H%M")
cfg_fn = pathlib.Path(f"{ts}.yaml")

cfg_fn.write_text(content, encoding="utf-8")

print("YAML saved to", cfg_fn)

try:
    cfg = yaml.safe_load(content)
except yaml.YAMLError as e:
    sys.exit(f"[ERROR] LLM output is not valid YAML:\n{e}")

dest_path = f"/workspace/DreamScene/configs/scenes/{cfg_fn.name}"
subprocess.run(["docker", "cp", str(cfg_fn), f"{args.container}:{dest_path}"], check=True)
print(f"Copied to container:{dest_path}")

run_cmd = (
    f"cd /workspace/DreamScene && "
    f"python main.py --config configs/scenes/{cfg_fn.name}"
)
print("DreamScene training started inside container …")
subprocess.run(["docker", "exec", "-it", args.container, "bash", "-c", run_cmd],
               check=True)
print("DreamScene training finished")

exp_name = cfg["log"]["exp_name"]
ply_in   = (f"/workspace/DreamScene/experiments/{exp_name}"
            f"/scene_checkpoints/scene_final_model.ply")
ply_out  = pathlib.Path("outputs") / f"{stamp}_scene.ply"
ply_out.parent.mkdir(exist_ok=True)

subprocess.run(["docker", "cp", f"{args.container}:{ply_in}", str(ply_out)], check=True)
print(f"Final PLY copied to host: {ply_out}")
