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
    generate_type: str = Literal[
        "t2v", "i2v", "v2v"
    ],  # i2v: image to video, v2v: video to video
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
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            model_path, torch_dtype=dtype
        )
        image = load_image(image=image_or_video_path)
    elif generate_type == "t2v":
        pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    else:
        pipe = CogVideoXVideoToVideoPipeline.from_pretrained(
            model_path, torch_dtype=dtype
        )
        video = load_video(image_or_video_path)

    # If you're using with lora, add this code
    if lora_path:
        pipe.load_lora_weights(
            lora_path,
            weight_name="pytorch_lora_weights.safetensors",
            adapter_name="test_1",
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

    class Args:
        prompt = None  # The description of the video to be generated
        image_or_video_path = (
            None  # The path of the image to be used as the background of the video
        )
        model_path = "THUDM/CogVideoX-5b"  # Path of the pre-trained model use
        lora_path = None  # The path of the LoRA weights to be used
        lora_rank = 128  # The rank of the LoRA weights
        output_path = None  # The path save generated video
        guidance_scale = 6.0  # The scale for classifier-free guidance
        num_inference_steps = 50  # Inference steps
        num_frames = 81  # Number of steps for the inference process
        width = None  # The width of the generated video
        height = None  # The height of the generated video
        fps = 16  # The frames per second for the generated video
        num_videos_per_prompt = 1  # Number of videos to generate per prompt
        generate_type = "t2v"  # The type of video generation
        dtype = "bfloat16"  # The data type for computation
        seed = None  # The seed for reproducibility

    args = Args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    ls_prompts = [
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He slides the product into his coat pocket.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He lowers the product to waist hip, and gradually tucks the product into his pants pocket.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He carefully and slowly slides the product into his back pocket.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He places the product into the inner-left pocket of his coat.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He lowers the product into his tote at hip level.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He places the product  into his shoulder bag at chest level.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He places the product  into his crossbody bag.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He places the product into his right shoulder bag with his right hand, while keeping his left hand steady.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He puts the product into a plastic bag hanging at the back of the cart.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He slips the product into the personal plastic bag held with his left hand.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He slips the product into the personal plastic bag hung from his left elbow in front of his chest.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He places the product into the bag slinging over his left shoulder.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He unwraps a cosmetics item to take out the product insides using both hands at hip height, and slides the product into his coat pocket.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He unwraps a cosmetics item to take out the product insides using both hands at hip height, and lowers the product to waist hip, and gradually tucks the product into his pants pocket.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He unwraps a cosmetics item to take out the product insides using both hands at hip height, and carefully and slowly slides the product into his back pocket.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He unwraps a cosmetics item to take out the product insides using both hands at hip height, and places the product into the inner-left pocket of his coat.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He unwraps a cosmetics item to take out the product insides using both hands at hip height, and lowers the product into his tote at hip level.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He unwraps a cosmetics item to take out the product insides using both hands at hip height, and places the product  into his shoulder bag at chest level.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He unwraps a cosmetics item to take out the product insides using both hands at hip height, and places the product  into his crossbody bag.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He unwraps a cosmetics item to take out the product insides using both hands at hip height, and places the product into his right shoulder bag with his right hand, while keeping his left hand steady.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He unwraps a cosmetics item to take out the product insides using both hands at hip height, and puts the product into a plastic bag hanging at the back of the cart.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He unwraps a cosmetics item to take out the product insides using both hands at hip height, and slips the product into the personal plastic bag held with his left hand.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He unwraps a cosmetics item to take out the product insides using both hands at hip height, and slips the product into the personal plastic bag hung from his left elbow in front of his chest.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He unwraps a cosmetics item to take out the product insides using both hands at hip height, and places the product into the bag slinging over his left shoulder.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He inspects a product on his right hand, and slips the product into the personal plastic bag held with his left hand.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He inspects a product on his right hand, and slips the product into the personal plastic bag hung from his left elbow in front of his chest.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He inspects a product on his right hand, and places the product into the bag slinging over his left shoulder.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He transfers a product from shelf into his coat pocket.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He transfers a product from shelf into his pants pocket.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He transfers a product from shelf into his back pocket.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He transfers a product from shelf into the inner-left pocket of his coat.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He transfers a product from shelf into his tote at hip level.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He transfers a product from shelf into his shoulder bag at chest level.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He transfers a product from shelf into his crossbody bag.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He transfers a product from shelf into his right shoulder bag with his right hand, while keeping his left hand steady.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He transfers a product from shelf into a plastic bag hanging at the back of the cart.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He transfers a product from shelf into the personal plastic bag held with his left hand.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He transfers a product from shelf into the personal plastic bag hung from his left elbow in front of his chest.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He transfers a product from shelf into the bag slinging over his left shoulder.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He transfers a product from the basket into his coat pocket.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He transfers a product from the basket into his pants pocket.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He transfers a product from the basket into his back pocket.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He transfers a product from the basket into the inner-left pocket of his coat.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He transfers a product from the basket into his tote at hip level.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He transfers a product from the basket into his shoulder bag at chest level.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He transfers a product from the basket into his crossbody bag.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He transfers a product from the basket into his right shoulder bag with his right hand, while keeping his left hand steady.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He transfers a product from the basket into a plastic bag hanging at the back of the cart.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He transfers a product from the basket into the personal plastic bag held with his left hand.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He transfers a product from the basket into the personal plastic bag hung from his left elbow in front of his chest.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He transfers a product from the basket into the bag slinging over his left shoulder.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He transfers a product from the basket held with his left hand into his coat pocket.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He transfers a product from the basket held with his left hand into his pants pocket.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He transfers a product from the basket held with his left hand into his back pocket.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He transfers a product from the basket held with his left hand into the inner-left pocket of his coat.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He transfers a product from the basket held with his left hand into his tote at hip level.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He transfers a product from the basket held with his left hand into his shoulder bag at chest level.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He transfers a product from the basket held with his left hand into his crossbody bag.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He transfers a product from the basket held with his left hand into his right shoulder bag with his right hand, while keeping his left hand steady.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He transfers a product from the basket held with his left hand into a plastic bag hanging at the back of the cart.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He transfers a product from the basket held with his left hand into the personal plastic bag held with his left hand.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He transfers a product from the basket held with his left hand into the personal plastic bag hung from his left elbow in front of his chest.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He transfers a product from the basket held with his left hand into the bag slinging over his left shoulder.",
        "A static surveillance camera captures a side-view, full-body view of a man with a shopping cart in a store aisle. He transfers a product from the shopping cart into his coat pocket.",
        "A static surveillance camera captures a side-view, full-body view of a man with a shopping cart in a store aisle. He transfers a product from the shopping cart into his pants pocket.",
        "A static surveillance camera captures a side-view, full-body view of a man with a shopping cart in a store aisle. He transfers a product from the shopping cart into his back pocket.",
        "A static surveillance camera captures a side-view, full-body view of a man with a shopping cart in a store aisle. He transfers a product from the shopping cart into the inner-left pocket of his coat.",
        "A static surveillance camera captures a side-view, full-body view of a man with a shopping cart in a store aisle. He transfers a product from the shopping cart into his tote at hip level.",
        "A static surveillance camera captures a side-view, full-body view of a man with a shopping cart in a store aisle. He transfers a product from the shopping cart into his shoulder bag at chest level.",
        "A static surveillance camera captures a side-view, full-body view of a man with a shopping cart in a store aisle. He transfers a product from the shopping cart into his crossbody bag.",
        "A static surveillance camera captures a side-view, full-body view of a man with a shopping cart in a store aisle. He transfers a product from the shopping cart into his right shoulder bag with his right hand, while keeping his left hand steady.",
        "A static surveillance camera captures a side-view, full-body view of a man with a shopping cart in a store aisle. He transfers a product from the shopping cart into a plastic bag hanging at the back of the cart.",
        "A static surveillance camera captures a side-view, full-body view of a man with a shopping cart in a store aisle. He transfers a product from the shopping cart into the personal plastic bag held with his left hand.",
        "A static surveillance camera captures a side-view, full-body view of a man with a shopping cart in a store aisle. He transfers a product from the shopping cart into the personal plastic bag hung from his left elbow in front of his chest.",
        "A static surveillance camera captures a side-view, full-body view of a man with a shopping cart in a store aisle. He transfers a product from the shopping cart into the bag slinging over his left shoulder.",
    ]
    args.output_path = "/home/lap_awlv/CogVideo/outputs/cogvideo_15/videos"
    args.num_frames = 49
    args.fps = 8

    for idx_prompt, prompt in enumerate(ls_prompts):
        print("\nPrompt {}: {}".format(idx_prompt, prompt))
        output_path = os.path.join(args.output_path, f"{idx_prompt}.mp4")
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
            num_videos_per_prompt=args.num_videos_per_prompt,
            dtype=dtype,
            generate_type=args.generate_type,
            seed=args.seed,
            fps=args.fps,
        )
        print(f"Video saved to {output_path}")
