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
$ python cli_demo.py --prompt "A girl riding a bike." --model_path THUDM/CogVideoX1.5-5b --generate_type "t2v"
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
            logging.warning(
                f"\033[1;31m{model_name} is not supported for custom resolution. Setting back to default resolution {desired_resolution}.\033[0m"
            )
            height, width = desired_resolution

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
        "A man stands in a busy market, his full body visible from a distance. He carefully picks up an item, examining it with interest. After glancing around to ensure no one is watching, he slowly tucks the item into his shoulder bag, adjusting the bag to secure it before moving on.",
        "From afar, a man in a bustling store holds an item in his hands, studying it closely. He checks his surroundings, then, with a measured motion, gently places the item into his bag, ensuring it fits snugly before continuing his shopping.",
        "A full-body view of a man shopping at a street fair. He lifts a unique item, admiring its details, then scans the area. With a cautious glance, he slowly tucks the item into his shoulder bag, making sure it’s well-hidden as he proceeds.",
        "Captured from a distance, a man examines an item in a vibrant market. He looks around cautiously, then with deliberate care, places the item into his bag, adjusting the strap to keep it secure as he glances back at the bustling crowd.",
        "A man is seen from a distance in a crowded shopping plaza. He holds a small item, taking a moment to appreciate it. After ensuring no one is observing him, he carefully places it into his shoulder bag, pulling the bag closed to conceal it.",
        "From a high angle, the camera captures a man shopping. He inspects an item with a focused gaze, then looks around. With a careful motion, he slips the item into his bag, adjusting it to ensure it remains secure as he blends into the crowd.",
        "A man is in a busy mall, viewed from a distance. He picks up an item, holding it thoughtfully while scanning the area. After a moment of consideration, he gently places it in his shoulder bag, making sure to zip it closed afterward.",
        "A full-body shot of a man at a flea market. He holds an interesting item, turning it over in his hands while observing other shoppers. After checking his surroundings, he slowly tucks it into his bag, adjusting the strap to keep it secure.",
        "From afar, a man is seen in a souvenir shop. He examines an intriguing item, looking around furtively, then carefully places it in his bag, ensuring he adjusts the zipper to keep it concealed.",
        "A man stands in a vibrant outdoor market, captured from a distance. He picks up a handcrafted item, taking a moment to appreciate its craftsmanship. After scanning the area, he carefully places it into his shoulder bag, making sure it’s snug.",
        "Captured from a distance, a man browses through a bookstore. He holds a rare book, checking to ensure no one is watching him. With a careful motion, he tucks the book into his bag, adjusting the strap for comfort.",
        "A full-body view of a man shopping for clothes. He holds a stylish shirt, examining it closely while observing the busy store. After ensuring no one is watching, he carefully places it in his shoulder bag, adjusting the bag as he moves.",
        "From a distance, a man is seen in a tech store. He picks up a gadget, glances around, and with a cautious motion, discreetly places it in his bag, ensuring it fits well before continuing his shopping.",
        "A man is shopping in a lively market, captured from a distance. He examines an item, checking for observers. After a moment of hesitation, he carefully places it in his shoulder bag, adjusting the strap to keep it secure.",
        "From afar, a man stands in a local grocery store. He picks up a specialty item, surveying the aisles. After a quick glance, he gently tucks it into his bag, ensuring it’s well-hidden as he moves through the store.",
        "A full-body view of a man in a busy shopping district. He holds an accessory, glancing over his shoulder. After checking for anyone watching, he carefully places it in his shoulder bag, securing it with a zip.",
        "Captured from a distance, a man inspects a collectible in a store. He looks around cautiously, then with deliberate care, places the item in his bag, ensuring it’s tucked away safely.",
        "A man is seen browsing in a crowded market, viewed from afar. He checks his surroundings, then holds an item thoughtfully before placing it in his shoulder bag, making sure it’s secure as he continues to shop.",
        "From a distance, a man stands in an outdoor craft fair. He admires an item, scanning the area for onlookers. With a careful motion, he places it into his bag, adjusting the strap for comfort.",
        "A man is shopping in a vibrant bazaar, captured from a distance. He holds a decorative item, glancing around to ensure no one is watching, and carefully puts it in his bag, adjusting it as he walks away.",
        "From afar, a man browses through a vintage shop. He observes an item closely, ensuring no one is nearby. After a moment of hesitation, he discreetly places it in his shoulder bag, securing it before leaving.",
        "A full-body view of a man in a bustling shopping center. He examines an item, looking around with intention. After ensuring he’s not being observed, he carefully places it in his bag, zipping it shut afterward.",
        "Captured from a distance, a man is seen in an art market. He holds a piece of artwork, glances around, and gently tucks it into his shoulder bag, ensuring it’s well-concealed.",
        "A man stands in a busy flea market, viewed from afar. He inspects an item, checking for observers. After a quick glance, he carefully places it in his bag, ensuring it’s secure as he continues shopping.",
        "From a distance, a man browses a local artisan shop. He holds a crafted piece, surveying the area before thoughtfully placing it into his shoulder bag.",
        "A full-body shot of a man shopping at a festival. He looks around while holding an item, then carefully places it in his bag, ensuring he blends in with the crowd.",
        "A man is seen from a distance in a vibrant shopping area. He checks his surroundings while holding something, then discreetly places it in his shoulder bag, adjusting it as he walks away.",
        "From afar, a man browses through a popular store. He glances around, then carefully adds something to his bag, ensuring it’s secure before continuing his shopping.",
        "A full-body view of a man at a street vendor. He scans the crowd while holding an item, then gently places it in his shoulder bag, making sure it’s well-hidden."
    ]
    for idx_prompt, prompt in enumerate(ls_prompts):
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
