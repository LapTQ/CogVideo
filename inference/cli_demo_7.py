"""
This script demonstrates how to generate a video using the CogVideoX model with the Hugging Face `diffusers` pipeline.
The script supports different types of video generation, including text-to-video (t2v), image-to-video (i2v),
and video-to-video (v2v), depending on the input data and different weight.

- text-to-video: THUDM/CogVideoX-5b, THUDM/CogVideoX-2b or THUDM/CogVideoX1.5-5b
- video-to-video: THUDM/CogVideoX-5b, THUDM/CogVideoX-2b or THUDM/CogVideoX1.5-5b
- image-to-video: THUDM/CogVideoX-5b-I2V or THUDM/CogVideoX1.5-5b-I2V

Running the Script:
To run the script, use the following command with appropriate arguments:

```bash
$ python cli_demo.py --prompt "A girl riding a bike. The camera is far away and static across frames, always capturing the whole body of the person in every frame. The camera is far away and static across frames, always capturing the whole body of the person in every frame.", --model_path THUDM/CogVideoX1.5-5b --generate_type "t2v"
```

You can change `pipe.enable_sequential_cpu_offload()` to `pipe.enable_model_cpu_offload()` to speed up inference, but this will use more GPU memory

Additional options are available to specify the model path, guidance scale, number of inference steps, video generation type, and output paths.

"""

import argparse
import logging
from typing import Literal, Optional
import os

import torch

from diffusers import (
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXPipeline,
    CogVideoXVideoToVideoPipeline,
)
from diffusers.utils import export_to_video, load_image, load_video


logging.basicConfig(level=logging.INFO)

# Recommended resolution for each model (width, height)
RESOLUTION_MAP = {
    # cogvideox1.5-*
    "cogvideox1.5-5b-i2v": (768, 1360),
    "cogvideox1.5-5b": (768, 1360),
    # cogvideox-*
    "cogvideox-5b-i2v": (480, 720),
    "cogvideox-5b": (480, 720),
    "cogvideox-2b": (480, 720),
}


def generate_video(
    prompt: str,
    model_path: str,
    lora_path: str = None,
    lora_rank: int = 128,
    num_frames: int = 81,
    width: Optional[int] = None,
    height: Optional[int] = None,
    output_path: str = "./output.mp4",
    image_or_video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal["t2v", "i2v", "v2v"],  # i2v: image to video, v2v: video to video
    seed: int = 42,
    fps: int = 16,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - num_frames (int): Number of frames to generate. CogVideoX1.0 generates 49 frames for 6 seconds at 8 fps, while CogVideoX1.5 produces either 81 or 161 frames, corresponding to 5 seconds or 10 seconds at 16 fps.
    - width (int): The width of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - height (int): The height of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - generate_type (str): The type of video generation (e.g., 't2v', 'i2v', 'v2v').·
    - seed (int): The seed for reproducibility.
    - fps (int): The frames per second for the generated video.
    """

    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
    # function to use Multi GPUs.

    image = None
    video = None

    model_name = model_path.split("/")[-1].lower()
    desired_resolution = RESOLUTION_MAP[model_name]
    if width is None or height is None:
        height, width = desired_resolution
        logging.info(
            f"\033[1mUsing default resolution {desired_resolution} for {model_name}\033[0m"
        )
    elif (height, width) != desired_resolution:
        if generate_type == "i2v":
            # For i2v models, use user-defined width and height
            logging.warning(
                f"\033[1;31mThe width({width}) and height({height}) are not recommended for {model_name}. The best resolution is {desired_resolution}.\033[0m"
            )
        else:
            # Otherwise, use the recommended width and height
            # logging.warning(
            #     f"\033[1;31m{model_name} is not supported for custom resolution. Setting back to default resolution {desired_resolution}.\033[0m"
            # )
            # height, width = desired_resolution
            pass

    if generate_type == "i2v":
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
        image = load_image(image=image_or_video_path)
    elif generate_type == "t2v":
        pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    else:
        pipe = CogVideoXVideoToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
        video = load_video(image_or_video_path)

    # If you're using with lora, add this code
    if lora_path:
        pipe.load_lora_weights(
            lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1"
        )
        pipe.fuse_lora(components=["transformer"], lora_scale=1.0)

    # 2. Set Scheduler.
    # Can be changed to `CogVideoXDPMScheduler` or `CogVideoXDDIMScheduler`.
    # We recommend using `CogVideoXDDIMScheduler` for CogVideoX-2B.
    # using `CogVideoXDPMScheduler` for CogVideoX-5B / CogVideoX-5B-I2V.

    # pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )

    # 3. Enable CPU offload for the model.
    # turn off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
    # and enable to("cuda")
    pipe.to("cuda")

    # pipe.enable_model_cpu_offload()
    # pipe.enable_sequential_cpu_offload()
    # pipe.vae.enable_slicing()
    # pipe.vae.enable_tiling()

    # 4. Generate the video frames based on the prompt.
    # `num_frames` is the Number of frames to generate.
    if generate_type == "i2v":
        video_generate = pipe(
            height=height,
            width=width,
            prompt=prompt,
            image=image,
            # The path of the image, the resolution of video will be the same as the image for CogVideoX1.5-5B-I2V, otherwise it will be 720 * 480
            num_videos_per_prompt=num_videos_per_prompt,  # Number of videos to generate per prompt
            num_inference_steps=num_inference_steps,  # Number of inference steps
            num_frames=num_frames,  # Number of frames to generate
            use_dynamic_cfg=True,  # This id used for DPM scheduler, for DDIM scheduler, it should be False
            guidance_scale=guidance_scale,
            # generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
        ).frames[0]
    elif generate_type == "t2v":
        video_generate = pipe(
            height=height,
            width=width,
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            # generator=torch.Generator().manual_seed(seed),
        ).frames[0]
    else:
        video_generate = pipe(
            height=height,
            width=width,
            prompt=prompt,
            video=video,  # The path of the video to be used as the background of the video
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            # generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
        ).frames[0]
    export_to_video(video_generate, output_path, fps=fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a video from a text prompt using CogVideoX"
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="The description of the video to be generated"
    )
    parser.add_argument(
        "--image_or_video_path",
        type=str,
        default=None,
        help="The path of the image to be used as the background of the video",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="THUDM/CogVideoX1.5-5B",
        help="Path of the pre-trained model use",
    )
    parser.add_argument(
        "--lora_path", type=str, default=None, help="The path of the LoRA weights to be used"
    )
    parser.add_argument("--lora_rank", type=int, default=128, help="The rank of the LoRA weights")
    parser.add_argument(
        "--output_path", type=str, default="./output.mp4", help="The path save generated video"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance"
    )
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Inference steps")
    parser.add_argument(
        "--num_frames", type=int, default=81, help="Number of steps for the inference process"
    )
    parser.add_argument("--width", type=int, default=None, help="The width of the generated video")
    parser.add_argument(
        "--height", type=int, default=None, help="The height of the generated video"
    )
    parser.add_argument(
        "--fps", type=int, default=16, help="The frames per second for the generated video"
    )
    parser.add_argument(
        "--num_videos_per_prompt",
        type=int,
        default=1,
        help="Number of videos to generate per prompt",
    )
    parser.add_argument(
        "--generate_type", type=str, default="t2v", help="The type of video generation"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="The data type for computation"
    )
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    ls_prompts = [
        "A man stands in the distance inside a store, visible full-body from a fixed camera. He picks up a product with one hand, glances around subtly, opens his shoulder bag with the other hand, and carefully slips the item inside while staying in place. The camera is far away and static across frames, always capturing the whole body of the person in every frame.",
    
        "In a store aisle, a man seen from a distance holds a product at chest level, looks side to side cautiously, then bends slightly to unzip his shoulder bag and gently insert the item inside. The camera remains static. The camera is far away and static across frames, always capturing the whole body of the person in every frame.",
        
        "A man positioned mid-frame in a fixed, distant shot looks at a product in his hand, squats slightly to reach into his bag, opens it wide, and slowly lowers the item in while scanning his surroundings. The camera is far away and static across frames, always capturing the whole body of the person in every frame.",
        
        "Filmed from a distant static angle, a man picks up a product, rotates slightly to face his bag, opens it with one hand, and places the item inside in a smooth, deliberate motion, maintaining his position. The camera is far away and static across frames, always capturing the whole body of the person in every frame.",
        
        "Seen from afar in a still camera frame, a man observes a product, turns his shoulder slightly, lifts the flap of his bag, and tucks the product in with care, glancing around once before closing the bag. The camera is far away and static across frames, always capturing the whole body of the person in every frame.",
        
        "A full-body view from a fixed camera shows a man holding a product at waist height, pausing briefly, then slowly placing the item into his shoulder bag, using both hands while staying in the same spot. The camera is far away and static across frames, always capturing the whole body of the person in every frame.",
        
        "In a wide shot, a man stands stationary. He examines a product, unzips his crossbody bag, shifts his weight to one leg, and slides the item in while keeping his posture relaxed. The camera is far away and static across frames, always capturing the whole body of the person in every frame.",
        
        "From a distant, stable viewpoint, a man steadies a product in one hand, adjusts the strap of his bag, flips it open, and places the item inside with practiced ease. The camera is far away and static across frames, always capturing the whole body of the person in every frame.",
        
        "Captured from afar, a man takes a product off a shelf, looks over his shoulder, carefully opens his shoulder bag, and slides the item in, all while staying in the same position. The camera is far away and static across frames, always capturing the whole body of the person in every frame.",
        
        "A man, fully visible from a distance in a static frame, holds a product for a moment, glances to both sides, then turns slightly and lowers the item into his shoulder bag deliberately. The camera is far away and static across frames, always capturing the whole body of the person in every frame.",
        
        "From a fixed camera angle, a man grabs a product, subtly checks his surroundings, shifts his torso to the left, opens his shoulder bag, and places the item inside with focused movements. The camera is far away and static across frames, always capturing the whole body of the person in every frame.",
        
        "A man in a store, fully visible from a static camera placed far away, holds a product close to his chest, turns his upper body slightly, opens his shoulder bag, and drops the item in with care. The camera is far away and static across frames, always capturing the whole body of the person in every frame.",
        
        "In a distant, unmoving shot, a man stands near a shelf, takes a product, opens his shoulder bag using one hand, and smoothly lowers the product inside with the other. The camera is far away and static across frames, always capturing the whole body of the person in every frame.",
        
        "The camera remains stationary as a man grabs a product, stands still, adjusts his bag, and slowly fits the item into the open bag while glancing discreetly around. The camera is far away and static across frames, always capturing the whole body of the person in every frame.",
        
        "Filmed from afar, a man inspects a product, then leans slightly to one side, opens his shoulder bag using both hands, and places the item inside without stepping away. The camera is far away and static across frames, always capturing the whole body of the person in every frame.",
        
        "A full-body distant view captures a man retrieving a product, then carefully adjusting his bag’s strap, opening it, and placing the item inside while standing in place. The camera is far away and static across frames, always capturing the whole body of the person in every frame.",
        
        "In a fixed wide shot, a man picks up a product, looks cautiously around, unzips his shoulder bag, and places the item in carefully, standing rooted in one spot. The camera is far away and static across frames, always capturing the whole body of the person in every frame.",
        
        "A static, far-shot view shows a man holding a product in both hands before sliding it carefully into his shoulder bag, never moving from his standing position. The camera is far away and static across frames, always capturing the whole body of the person in every frame.",
        
        "From a fixed, distant camera angle, a man lifts a product to inspect it, holds it in one hand while opening his bag, and places it inside, remaining stationary. The camera is far away and static across frames, always capturing the whole body of the person in every frame.",
        
        "Seen from a far static camera, a man grabs a product, examines it briefly, then shifts slightly to unzip his shoulder bag and insert the item carefully inside. The camera is far away and static across frames, always capturing the whole body of the person in every frame.",
        
        "A man seen fully in a distant wide shot takes a product from a shelf, holds it momentarily, opens his bag, and smoothly places the product in, remaining in place. The camera is far away and static across frames, always capturing the whole body of the person in every frame.",
        
        "Captured from a far, motionless angle, a man lifts a product, leans a little, opens his shoulder bag flap with one hand, and places the product in while standing still. The camera is far away and static across frames, always capturing the whole body of the person in every frame.",
        
        "A man fully in frame from a distance holds a product, slowly turns his upper body to the left, and places the item inside his shoulder bag in a calm motion. The camera is far away and static across frames, always capturing the whole body of the person in every frame.",
        
        "In a static full-body shot, a man stands firm as he carefully inspects a product, loosens the flap of his shoulder bag, and gently puts the item inside. The camera is far away and static across frames, always capturing the whole body of the person in every frame.",
        
        "From a wide unmoving frame, a man stands near a shelf, lifts a product with care, adjusts his bag strap, opens the bag, and lowers the item in slowly. The camera is far away and static across frames, always capturing the whole body of the person in every frame.",
        
        "A man, captured entirely from a distant fixed camera, holds a product mid-air, glances over his shoulder, and places it precisely into his open shoulder bag. The camera is far away and static across frames, always capturing the whole body of the person in every frame.",
        
        "In a long shot with no camera movement, a man calmly retrieves a product, checks his surroundings, and lowers the item into his shoulder bag without leaving his spot. The camera is far away and static across frames, always capturing the whole body of the person in every frame.",
        
        "A far shot shows a man taking a product, shifting his weight, opening his bag with a practiced motion, and slipping the product inside while staying put. The camera is far away and static across frames, always capturing the whole body of the person in every frame.",
        
        "In a still wide frame, a man reaches for a product, inspects it, slowly opens his shoulder bag with one hand, and carefully inserts the item while standing steady. The camera is far away and static across frames, always capturing the whole body of the person in every frame.",
        
        "From a distant static view, a man stands upright, turns slightly, and with fluid motion places a product into his bag, never changing location. The camera is far away and static across frames, always capturing the whole body of the person in every frame.",
    ]
    for idx_prompt, prompt in enumerate(ls_prompts):
        print("\nPrompt {}: {}".format(idx_prompt, prompt))
        for idx_video in range(args.num_videos_per_prompt):
            output_path = args.output_path.format(idx_prompt, idx_video)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            generate_video(
                prompt=prompt,
                model_path=args.model_path,
                lora_path=args.lora_path,
                lora_rank=args.lora_rank,
                output_path=output_path,
                num_frames=args.num_frames,
                width=args.width,
                height=args.height,
                image_or_video_path=args.image_or_video_path,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                num_videos_per_prompt=1,
                dtype=dtype,
                generate_type=args.generate_type,
                seed=args.seed,
                fps=args.fps,
            )
            print(f"Video saved to {output_path}")
