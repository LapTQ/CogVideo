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

    # using Gemini
    ls_prompts = [
        # --- Group 1: Coat / Jacket Pockets ---
        "A static surveillance camera films a high-angle, wide shot of a person standing in a store aisle, front-facing. They *quickly* examine a product, then *swiftly* slide it into their right coat pocket with a suspicious glance.",
        "A fixed, high-angle camera provides a wide, side-view shot of a person in an aisle. They *hastily* look at a small item before *abruptly* slipping it into the exterior pocket of their jacket.",
        "From a ceiling-mounted perspective, a wide, front-view shot shows a person *rapidly* lowering an object and *furtively* tucking it into the inner lining of their open jacket.",
        "A static, high-angle camera captures a wide shot, front-view, of a person *suddenly* placing a product from the shelf into their left jacket pocket with their left hand.",
        "A wide, high-angle surveillance shot captures a side-view of a person. They hold an item in their right hand and *swiftly* transfer it *discreetly* into their left inner coat pocket.",
        "Static high-angle wide shot, front view. A person in a retail aisle pauses *briefly*, then *quickly* inserts a small, boxed item into their coat pocket at waist level with a suspicious motion.",
        "Fixed camera, elevated perspective, wide shot. A side-view of a person shows them *hastily* placing an object into the breast pocket of their jacket.",
        "A high-angle, static surveillance camera captures a wide, front-facing view of a person who *snatches* an item and *abruptly* inserts it into their jacket's side pocket.",
        "Ceiling-level camera, static wide shot, side-view. A person angles their body *sharply* and *quickly* slips a product into the inner pocket of their blazer.",
        "A fixed high-angle wide shot shows a front-view of a person *quickly* placing an item into their right jacket pocket, using their right hand with a *furtive* glance.",
        # --- Group 2: Pants Pockets ---
        "A static high-angle surveillance shot captures a front-facing person in a store aisle. They *quickly* inspect a product, lower it to hip level *abruptly*, and *swiftly* tuck it into their front-right pants pocket.",
        "A fixed camera from a high-angle, side-view, captures a wide shot of a person *rapidly* moving a product from the shelf directly into their back-right pants pocket with a suspicious movement.",
        "Ceiling-mounted camera, static wide shot, front-view. A person holds an item, *quickly* looks around, and then *furtively* pushes it down into their front-left pants pocket.",
        "A high-angle, wide surveillance shot shows a side-view of a person. They *hastily* take an item from a shelf and *abruptly* insert it into their back-left pants pocket.",
        "Static, high-angle camera, wide shot, front-view. A person lowers an item below waist level *suddenly* and *quickly* slips it into their right pants pocket.",
        "A fixed, elevated camera captures a wide, side-view of a person *quickly* bending *furtively* to insert an object into their cargo pants' side pocket.",
        "A static high-angle wide shot, front-view, shows a person *swiftly* transferring an item from their hand into their front-left pants pocket using their left hand with a suspicious glance.",
        "From a high-angle, a static wide shot captures a side-view of a person *quickly* placing a cylindrical item into their back pocket with a hurried motion.",
        "Ceiling camera, wide shot, front-view. A person standing in an aisle *rapidly* moves an item from shelf-height down to their hip and *abruptly* into their right pocket.",
        "Static surveillance camera, high-angle wide shot, side-view. A person *briefly* examines an item before *quickly* sliding it into their front pants pocket with a suspicious manner.",
        # --- Group 3: Bags - Tote/Shoulder/Crossbody ---
        "A static ceiling-mounted camera records a front-view, wide shot of a person in a store aisle *quickly* lowering a product into their open tote bag at hip height with a hurried movement.",
        "A fixed surveillance camera captures a high-angle wide shot of a person in a store aisle *swiftly* placing a product into their shoulder bag at chest level.",
        "A fixed high-angle surveillance camera captures a side-view wide shot of a person in a store aisle *abruptly* placing a product into their crossbody bag with a suspicious glance.",
        "A high-angle, wide shot from a static camera shows a front-view of a person *quickly* opening their messenger bag flap and *rapidly* dropping an item inside.",
        "Static high-angle wide shot, side-view. A person holding a shoulder bag open with one hand *hastily* uses their other hand to place an item inside it.",
        "A static camera from a high-angle captures a wide, front-view of a person *quickly* slipping an item into a large tote bag resting on the floor by their feet with a furtive motion.",
        "Fixed, high-angle camera, wide shot, side-view. A person *rapidly* moves an item from the shelf and *abruptly* places it into a canvas tote bag hanging from their shoulder.",
        "A ceiling-mounted camera provides a wide, front-view shot. A person *quickly* uses their right hand to place an item into a crossbody bag worn on their left hip with a suspicious glance.",
        "Static surveillance camera, high-angle wide shot, side-view. A person holds an open shoulder bag and *hastily* drops a small product inside before *quickly* closing it.",
        "A wide, high-angle shot, front-view. A person *briefly* opens their messenger bag, *swiftly* inserts an item, and *quickly* fastens the flap with a hurried motion.",
        "Fixed camera, elevated perspective, wide shot, side-view. A person *quickly* places an item from the shelf into an open tote bag hanging on their forearm with a suspicious movement.",
        "Static, high-angle, wide-angle front view. A person *quickly* adjusts the strap of their shoulder bag and *swiftly* slides an object inside it.",
        "A high-angle, wide shot from a static camera captures a side-view of a person *rapidly* placing an item into a large, unzipped shoulder bag.",
        "Ceiling-mounted camera, wide shot, front-view. A person *quickly* lowers an item into a tote bag held in their left hand with a suspicious glance.",
        "A fixed surveillance camera captures a wide, high-angle side-view of a person *swiftly* placing an item into a small handbag hanging from their wrist.",
        # --- Group 4: Bags - Backpacks ---
        "A static, high-angle surveillance camera films a wide, side-view shot of a person *quickly* turned, *hastily* unzipping a backpack side pocket and *abruptly* inserting an item.",
        "A fixed high-angle wide shot, front-view. A person *quickly* takes off their backpack, places it on the floor, *swiftly* unzips the main compartment, and *rapidly* places an item inside.",
        "From a ceiling-mounted perspective, a wide, side-view shot shows a person, wearing a backpack on their front, *quickly* placing an item into the open top pouch with a suspicious glance.",
        "A static high-angle wide shot, side-view. A person *rapidly* reaches back to their backpack, *quickly* opens a zipper, and *swiftly* slips an item inside without removing the bag.",
        "A fixed, elevated camera captures a wide, front-view of a person holding a backpack, *hastily* opening a front pouch, and *abruptly* placing a small item within it.",
        # --- Group 5: Shopping Cart / Basket ---
        "A high-angle surveillance camera captures a side-view wide shot of a person in a store aisle *quickly* putting a product inside a plastic bag hanging from the back of the shopping cart with a suspicious movement.",
        "A static surveillance camera captures a high-angle front-view wide shot of a person *swiftly* transferring a product from the handheld basket in their left hand into their right coat pocket.",
        "A fixed, ceiling-mounted camera shows a wide, front-view of a person *rapidly* moving an item from their shopping basket and *quickly* placing it into a tote bag sitting inside the cart.",
        "A high-angle, wide shot, side-view. A person *hastily* takes an item from a shelf and *abruptly* places it into a reusable bag already positioned inside the main part of the shopping cart.",
        "Static camera, high-angle wide shot, front-view. A person holding a red shopping basket *quickly* lowers an item from the basket into their shoulder bag with a suspicious glance.",
        "A fixed, high-angle camera captures a wide, side-view of a person *swiftly* placing an item from the shelf underneath other items in their shopping cart.",
        "Ceiling-level camera, static wide shot, front-view. A person *rapidly* moves an item from their handheld basket and *quickly* slips it into their jacket pocket.",
        "A high-angle, static surveillance camera captures a wide, side-view of a person *hastily* transferring an item from their shopping cart and *abruptly* placing it into a backpack they are carrying.",
        "A fixed high-angle wide shot, front-view, shows a person *quickly* placing an item into a plastic bag that is hanging on the shopping cart's handle with a suspicious movement.",
        "Static, elevated camera, wide side-view. A person *rapidly* takes an item from their basket on the floor and *swiftly* places it into their pants pocket.",
        # --- Group 6: Package Manipulation ---
        "A static surveillance camera captures a high-angle front-view wide shot of a person in a store aisle *quickly* unwrapping a cosmetics package at hip height, *hastily* removing the inner product, and *swiftly* slipping it into a small bag.",
        "A fixed, high-angle wide shot, side-view. A person *abruptly* tears open a cardboard box, *rapidly* removes the contents, and *quickly* places the inner item into their coat pocket with a suspicious glance.",
        "Ceiling-mounted camera, wide front-view. A person *swiftly* uses both hands to open a blister pack, *hastily* extracts the small item, and immediately *quickly* closes their hand around it, lowering it to their pocket.",
        "A static high-angle camera, wide side-view. A person *quickly* removes an item from its packaging and *swiftly* places the item into their shoulder bag, leaving the empty package on the shelf with a hurried motion.",
        "A wide, high-angle surveillance shot, front-view. A person *rapidly* manipulates a package at waist level, *hastily* removes the inner product, and *swiftly* inserts it into their front pants pocket.",
        "Fixed camera, elevated perspective, wide shot, side-view. A person is seen *quickly* opening a box, *rapidly* taking the item out, and *swiftly* placing the item into a tote bag with a suspicious movement.",
        "A static high-angle wide shot, front-view, shows a person *hastily* unwrapping an item and *quickly* placing the unwrapped item into their jacket, while dropping the wrapper on the floor.",
        "From a high-angle, a static wide shot captures a side-view of a person *rapidly* removing a product from its outer packaging and *swiftly* slipping it into their back pocket with a hurried motion.",
        "Ceiling camera, wide shot, front-view. A person *quickly* opens a small box, *hastily* removes the item, and *swiftly* transfers it to their crossbody bag.",
        "A static surveillance camera, high-angle wide shot, side-view. A person *rapidly* removes an item from a package and *quickly* places the item into a bag hanging on their shopping cart with a suspicious glance.",
    ]
    args.output_path = "/home/lap_awlv/CogVideo/outputs/cogvideo_13/videos"
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
