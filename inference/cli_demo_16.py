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
        "The static surveillance camera provides a side-view, full-body shot of a man in a store aisle as he slips an item into his coat pocket.",
        "A full-body, side-facing image from a fixed security camera shows a man in an aisle lowering an item to his hip before gradually tucking it into his pants pocket.",
        "A man in a store aisle is seen in a static, side-view, full-body capture carefully and slowly sliding a product into his back pocket.",
        "From a static surveillance camera's side-view, full-body perspective, a man in an aisle places the merchandise into the inner-left pocket of his coat.",
        "The static surveillance footage shows a full-body, side-view of a man in an aisle putting the product into his tote bag at hip level.",
        "A fixed security camera captures a full-body, side-view of a man in a store aisle putting the item into his shoulder bag at chest height.",
        "A side-view, full-body image from a static surveillance camera captures a man in an aisle placing the product into his crossbody bag.",
        "In a static, side-view, full-body recording, a man in a store aisle uses his right hand to place the product into his right shoulder bag, while his left hand remains steady.",
        "A fixed surveillance camera records a full-body, side-view of a man in an aisle putting the product into a plastic bag hanging from the back of the shopping cart.",
        "The static security camera captures a side-view, full-body shot of a man in a store aisle slipping the product into the plastic bag he holds with his left hand.",
        "A full-body, side-facing shot from a static surveillance camera shows a man in an aisle slipping the item into the plastic bag hung from his left elbow in front of his chest.",
        "From a static surveillance camera's side-view, full-body perspective, a man in an aisle places the product into the bag slung over his left shoulder.",
        "The static camera footage captures a man in an aisle unwrapping a cosmetics item with both hands at hip height to retrieve the product inside, which he then slides into his coat pocket.",
        "A man in a store aisle is recorded by a static camera unwrapping a cosmetic item with both hands at hip level, taking out the product, lowering it to his hip, and gradually tucking it into his pants pocket.",
        "A static, side-view, full-body shot shows a man in an aisle unwrapping a cosmetics item at hip height to get the product, which he then carefully and slowly slides into his back pocket.",
        "In a full-body, side-view capture, a man in a store aisle unwraps a cosmetics item with both hands at hip level, takes out the contents, and places the product into the inner-left pocket of his coat.",
        "A static surveillance camera captures a man in an aisle unwrapping a cosmetics item with both hands at hip height, taking the product out, and lowering it into his tote bag at hip level.",
        "A full-body, side-view image from a static security camera captures a man in a store aisle concealing an item in his coat pocket.",
        "A man in a store aisle is shown in a static, full-body, side-facing shot, discreetly tucking an item into his pants pocket after lowering it to his hip.",
        "A side-view, full-body capture from a fixed camera shows a man in a store aisle slowly and carefully sliding a product into his back pocket.",
        "A man in an aisle is seen from a static surveillance camera's side-view, full-body perspective, placing merchandise into his coat's inner-left pocket.",
        "Static surveillance footage provides a full-body, side-view of a man in an aisle putting a product into his tote bag at hip level.",
        "A fixed security camera captures a full-body, side-view of a man in an aisle placing an item into his shoulder bag at chest height.",
        "A man in an aisle is captured by a static surveillance camera's side-view, full-body image, putting a product into his crossbody bag.",
        "A static, full-body, side-view recording shows a man in a store aisle using his right hand to place a product into his right shoulder bag while his left hand remains still.",
        "A fixed surveillance camera records a full-body, side-view of a man in an aisle placing a product into a plastic bag hanging from the back of his shopping cart.",
        "The static security camera captures a full-body, side-view of a man in a store aisle slipping a product into the plastic bag he is holding with his left hand.",
        "A full-body, side-facing shot from a static surveillance camera shows a man in an aisle slipping an item into the plastic bag hung from his left elbow in front of his chest.",
        "A man in an aisle is seen from a static surveillance camera's side-view, full-body perspective, placing a product into the bag slung over his left shoulder.",
        "Static camera footage captures a man in an aisle using both hands at hip height to unwrap a cosmetics item, retrieve the product inside, and then slide it into his coat pocket.",
        "A man in a store aisle is recorded by a static camera unwrapping a cosmetic item with both hands at hip level, removing the product, lowering it to his hip, and gradually tucking it into his pants pocket.",
        "A static, full-body, side-view shot shows a man in an aisle unwrapping a cosmetics item at hip height, taking out the product, and then slowly and carefully sliding it into his back pocket.",
        "In a full-body, side-view capture, a man in a store aisle unwraps a cosmetics item with both hands at hip level, takes out the contents, and places the product into the inner-left pocket of his coat.",
        "A static surveillance camera captures a man in an aisle unwrapping a cosmetics item with both hands at hip height, taking the product out, and lowering it into his tote bag at hip level.",
        "The full-body, side-view static camera shows a man in an aisle unwrapping a cosmetics item at hip height, retrieving the product, and placing it into his shoulder bag at chest level.",
        "A man in a store aisle is recorded by a static camera unwrapping a cosmetic item with both hands at hip height, taking out the product, and placing it into his crossbody bag.",
        "The static surveillance footage captures a man in an aisle unwrapping a cosmetics item at hip height, retrieving the product, and using his right hand to place it into his right shoulder bag while keeping his left hand steady.",
        "A full-body, side-view static camera captures a man in a store aisle unwrapping a cosmetics item with both hands at hip height, removing the product, and putting it into a plastic bag hanging at the back of the cart.",
        "A man in an aisle is seen in a static, side-view, full-body shot unwrapping a cosmetics item with both hands at hip height, taking out the product, and slipping it into the personal plastic bag held in his left hand.",
        "A static surveillance camera records a full-body, side-view of a man in an aisle unwrapping a cosmetics item at hip height, taking out the product, and slipping it into the personal plastic bag hung from his left elbow in front of his chest.",
        "In a full-body, side-view capture, a man in a store aisle unwraps a cosmetics item at hip height, retrieves the product, and places it into the bag slinging over his left shoulder.",
        "The static surveillance camera provides a side-view, full-body shot of a man in a store aisle inspecting a product in his right hand before slipping it into the plastic bag held in his left hand.",
        "A full-body, side-facing shot from a static surveillance camera shows a man in an aisle inspecting an item in his right hand before slipping it into the plastic bag hung from his left elbow in front of his chest.",
        "In a static, side-view, full-body recording, a man in a store aisle inspects a product in his right hand before placing it into the bag slinging over his left shoulder.",
        "The static surveillance camera provides a side-view, full-body shot of a man in a store aisle moving a product from the shelf into his coat pocket.",
        "A full-body, side-facing image from a fixed security camera shows a man in an aisle moving an item from the shelf into his pants pocket.",
        "A man in a store aisle is seen in a static, side-view, full-body capture moving a product from the shelf into his back pocket.",
        "From a static surveillance camera's side-view, full-body perspective, a man in an aisle moves an item from the shelf into the inner-left pocket of his coat.",
        "The static surveillance footage shows a full-body, side-view of a man in an aisle moving a product from the shelf into his tote bag at hip level.",
        "A fixed security camera captures a full-body, side-view of a man in a store aisle moving a product from the shelf into his shoulder bag at chest height.",
        "A side-view, full-body image from a static surveillance camera captures a man in an aisle moving a product from the shelf into his crossbody bag.",
        "A man in a store aisle is captured by a static camera, unwrapping a cosmetics item near his hips, removing the product, and putting it into his shoulder bag near his chest.",
        "Static camera footage shows a man in a store aisle unwrapping a cosmetic item with both hands at hip level, taking the product out, and placing it into his crossbody bag.",
        "Surveillance footage from a static camera shows a man in an aisle unwrapping a cosmetics item at hip height, retrieving the product, and using his right hand to put it into his right shoulder bag.",
        "A static, full-body side-view camera records a man in a store aisle unwrapping a cosmetics item with both hands at hip level, removing the product, and putting it into a plastic bag hanging on the back of his cart.",
        "A man is seen in a static, side-view, full-body shot in an aisle, unwrapping a cosmetics item with both hands at hip height, taking out the product, and slipping it into a personal plastic bag held in his left hand.",
        "A static surveillance camera captures a full-body side-view of a man in an aisle unwrapping a cosmetics item at hip height, removing the product, and slipping it into a personal plastic bag hung from his left elbow.",
        "A full-body, side-view capture shows a man in a store aisle unwrapping a cosmetics item at hip height, retrieving the product, and placing it into the bag slung over his left shoulder.",
        "A static surveillance camera provides a side-view, full-body shot of a man in a store aisle inspecting a product in his right hand before slipping it into the plastic bag held in his left hand.",
        "A full-body, side-facing shot from a fixed security camera shows a man in an aisle examining an item in his right hand before slipping it into the plastic bag hung from his left elbow.",
        "A static, side-view, full-body recording shows a man in a store aisle inspecting a product in his right hand before placing it into the bag slung over his left shoulder.",
        "Static surveillance camera footage provides a side-view, full-body image of a man in an aisle moving a product from the shelf into his coat pocket.",
        "A fixed security camera captures a full-body, side-facing view of a man in an aisle moving an item from the shelf into his pants pocket.",
        "A man is seen in a static, side-view, full-body capture in a store aisle, moving a product from the shelf into his back pocket.",
        "From a static surveillance camera's side-view, full-body perspective, a man in an aisle moves an item from the shelf into the inner-left pocket of his coat.",
        "Static surveillance footage shows a full-body, side-view of a man in an aisle moving a product from the shelf into his tote bag at hip level.",
        "A fixed security camera captures a full-body, side-view of a man in a store aisle moving a product from the shelf into his shoulder bag near his chest.",
        "A side-view, full-body image from a static surveillance camera captures a man in an aisle moving a product from the shelf into his crossbody bag.",
        "In a static, side-view, full-body recording, a man in a store aisle uses his right hand to transfer a product from the shelf into his right shoulder bag, while his left hand remains steady.",
        "A fixed surveillance camera records a full-body, side-view of a man in an aisle moving a product from the shelf into a plastic bag hanging from the back of the cart.",
        "The static security camera captures a side-view, full-body shot of a man in a store aisle transferring a product from the shelf into the plastic bag held with his left hand.",
        "A full-body, side-facing shot from a static surveillance camera shows a man in an aisle moving a product from the shelf into the plastic bag hung from his left elbow in front of his chest.",
        "From a static surveillance camera's side-view, full-body perspective, a man in an aisle moves a product from the shelf into the bag slung over his left shoulder.",
        "The static surveillance camera provides a side-view, full-body shot of a man in a store aisle taking an item from the basket and putting it into his coat pocket.",
        "A full-body, side-facing image from a fixed security camera shows a man in an aisle moving an item from the basket into his pants pocket.",
        "A man in a store aisle is seen in a static, side-view, full-body capture moving a product from the basket into his back pocket.",
        "From a static surveillance camera's side-view, full-body perspective, a man in an aisle transfers an item from the basket into the inner-left pocket of his coat.",
        "The static surveillance footage shows a full-body, side-view of a man in an aisle moving a product from the basket into his tote bag at hip level.",
        "A fixed security camera captures a full-body, side-view of a man in a store aisle transferring a product from the basket into his shoulder bag at chest height.",
        "A side-view, full-body image from a static surveillance camera captures a man in an aisle moving a product from the basket into his crossbody bag.",
        "In a static, side-view, full-body recording, a man in a store aisle uses his right hand to transfer a product from the basket into his right shoulder bag, while his left hand remains steady.",
        "A fixed surveillance camera records a full-body, side-view of a man in an aisle moving a product from the basket into a plastic bag hanging from the back of the cart.",
        "The static security camera captures a side-view, full-body shot of a man in a store aisle transferring a product from the basket into the plastic bag held with his left hand.",
        "A full-body, side-facing shot from a static surveillance camera shows a man in an aisle moving a product from the basket into the plastic bag hung from his left elbow in front of his chest.",
        "From a static surveillance camera's side-view, full-body perspective, a man in an aisle moves a product from the basket into the bag slung over his left shoulder.",
        "The static surveillance camera provides a side-view, full-body shot of a man in a store aisle taking a product from the basket held in his left hand and putting it into his coat pocket.",
        "A full-body, side-facing image from a fixed security camera shows a man in an aisle moving an item from the basket held in his left hand into his pants pocket.",
        "A static, side-view camera records a man in a store aisle placing a product from a shelf into his right shoulder bag using his right hand, while his left hand is still.",
        "Fixed surveillance footage shows a man from the side, full-body, in an aisle transferring a product from the shelf into a plastic bag attached to the back of his shopping cart.",
        "A man in a store aisle is captured full-body and side-view by a static security camera as he moves a product from the shelf into a plastic bag he holds with his left hand.",
        "A full-body, side shot from a stationary surveillance camera depicts a man in an aisle shifting a product from the shelf into a plastic bag draped over his left elbow in front of his chest.",
        "The static surveillance camera's side, full-body perspective shows a man in an aisle putting a product from the shelf into a bag hanging from his left shoulder.",
        "Static surveillance footage provides a side, full-body view of a man in a store aisle taking an item out of a basket and tucking it into his coat pocket.",
        "A fixed security camera, in a full-body, side-facing image, captures a man in an aisle moving an item from the basket into his pants pocket.",
        "A man in a store aisle, seen in a static, side, full-body shot, moves a product from the basket into his back pocket.",
        "From a static surveillance camera's side, full-body viewpoint, a man in an aisle transfers an item from the basket into the inner-left pocket of his coat.",
        "The static surveillance footage provides a side, full-body view of a man in an aisle moving a product from the basket into his hip-level tote bag.",
        "A fixed security camera captures a side, full-body view of a man in a store aisle moving a product from the basket into his shoulder bag, which is at chest height.",
        "A static surveillance camera provides a side, full-body image of a man in an aisle putting a product from the basket into his crossbody bag.",
        "In a static, side, full-body recording, a man in a store aisle places a product from the basket into his right shoulder bag with his right hand, keeping his left hand steady.",
        "A fixed surveillance camera records a man full-body and side-view in an aisle, moving a product from the basket into a plastic bag hanging from the back of the cart.",
        "A static security camera captures a side, full-body shot of a man in a store aisle transferring a product from the basket into a plastic bag held by his left hand.",
        "A full-body, side shot from a static surveillance camera shows a man in an aisle moving a product from the basket into a plastic bag slung from his left elbow in front of his chest.",
        "From a static surveillance camera's side, full-body perspective, a man in an aisle moves a product from the basket into a bag draped over his left shoulder.",
        "The static surveillance camera provides a side, full-body shot of a man in a store aisle taking a product from the basket in his left hand and putting it into his coat pocket.",
        "A full-body, side-facing image from a fixed security camera shows a man in an aisle moving an item from the basket in his left hand into his pants pocket.",
        "A man in a store aisle is seen in a static, side-view, full-body capture moving a product from the basket held in his left hand into his back pocket.",
        "From a static surveillance camera's side-view, full-body perspective, a man in an aisle transfers an item from the basket held in his left hand into the inner-left pocket of his coat.",
        "The static surveillance footage shows a full-body, side-view of a man in an aisle moving a product from the basket held in his left hand into his tote bag at hip level.",
        "A fixed security camera captures a full-body, side-view of a man in a store aisle transferring a product from the basket held in his left hand into his shoulder bag at chest height.",
        "A side-view, full-body image from a static surveillance camera captures a man in an aisle moving a product from the basket held in his left hand into his crossbody bag.",
        "In a static, side-view, full-body recording, a man in a store aisle uses his right hand to transfer a product from the basket held in his left hand into his right shoulder bag, while his left hand remains steady.",
        "A fixed surveillance camera records a full-body, side-view of a man in an aisle moving a product from the basket held in his left hand into a plastic bag hanging at the back of the cart.",
        "The static security camera captures a side-view, full-body shot of a man in a store aisle transferring a product from the basket held in his left hand into the plastic bag he holds with his left hand.",
        "A full-body, side-facing shot from a static surveillance camera shows a man in an aisle moving a product from the basket held in his left hand into the plastic bag hung from his left elbow in front of his chest.",
        "From a static surveillance camera's side-view, full-body perspective, a man in an aisle moves a product from the basket held in his left hand into the bag slung over his left shoulder.",
        "The static surveillance camera provides a side-view, full-body shot of a man with a shopping cart in a store aisle moving a product from the cart into his coat pocket.",
        "A full-body, side-facing image from a fixed security camera shows a man with a shopping cart in an aisle transferring an item from the cart into his pants pocket.",
        "A man with a shopping cart in a store aisle is seen in a static, side-view, full-body capture moving a product from the cart into his back pocket.",
        "From a static surveillance camera's side-view, full-body perspective, a man with a cart in an aisle transfers an item from the cart into the inner-left pocket of his coat.",
        "The static surveillance footage shows a full-body, side-view of a man with a shopping cart in an aisle moving a product from the cart into his tote bag at hip level.",
        "A fixed security camera captures a full-body, side-view of a man with a shopping cart in a store aisle transferring a product from the cart into his shoulder bag at chest height.",
        "A side-view, full-body image from a static surveillance camera captures a man with a shopping cart in an aisle moving a product from the cart into his crossbody bag.",
        "In a static, side-view, full-body recording, a man with a shopping cart in a store aisle uses his right hand to transfer a product from the cart into his right shoulder bag, while his left hand remains steady.",
        "A man in a store aisle is captured in a static, side-view, full-body shot, placing an item from the basket in his left hand into his back pocket.",
        "Static side-view, full-body surveillance footage shows a man in an aisle shifting an item from the basket in his left hand to the inner-left pocket of his coat.",
        "The static surveillance video, from a full-body, side-view, depicts a man in an aisle moving a product from the basket he holds in his left hand into his hip-level tote bag.",
        "A fixed security camera records a full-body, side-view of a man in a store aisle putting a product from the basket in his left hand into his shoulder bag, which is at chest height.",
        "Static surveillance camera footage, side-view and full-body, shows a man in an aisle moving an item from the basket in his left hand into his crossbody bag.",
        "A static, side-view, full-body recording captures a man in a store aisle using his right hand to move a product from the basket in his left hand into his right shoulder bag, keeping his left hand steady.",
        "A fixed surveillance camera records a full-body, side-view of a man in an aisle placing a product from the basket in his left hand into a plastic bag hanging on the back of the cart.",
        "The static security camera provides a side-view, full-body shot of a man in a store aisle transferring a product from the basket in his left hand into the plastic bag he also holds with his left hand.",
        "From a static surveillance camera, a full-body, side-facing shot shows a man in an aisle moving a product from the basket in his left hand into a plastic bag hung from his left elbow near his chest.",
        "Static surveillance footage, side-view and full-body, captures a man in an aisle shifting a product from the basket in his left hand into the bag slung over his left shoulder.",
        "The static surveillance camera provides a side-view, full-body shot of a man with a shopping cart in a store aisle placing a product from the cart into his coat pocket.",
        "A fixed security camera's full-body, side-facing image shows a man with a shopping cart in an aisle transferring an item from the cart into his pants pocket.",
        "A man with a shopping cart in a store aisle is captured in a static, side-view, full-body shot, moving a product from the cart into his back pocket.",
        "Static side-view, full-body surveillance footage shows a man with a cart in an aisle shifting an item from the cart to the inner-left pocket of his coat.",
        "The static surveillance video, from a full-body, side-view, depicts a man with a shopping cart in an aisle moving a product from the cart into his hip-level tote bag.",
        "A fixed security camera records a full-body, side-view of a man with a shopping cart in a store aisle putting a product from the cart into his shoulder bag, which is at chest height.",
        "Static surveillance camera footage, side-view and full-body, shows a man with a shopping cart in an aisle moving an item from the cart into his crossbody bag.",
        "A static, side-view, full-body recording captures a man with a shopping cart in a store aisle using his right hand to move a product from the cart into his right shoulder bag, keeping his left hand steady.",
        "A fixed surveillance camera records a full-body, side-view of a man with a shopping cart in an aisle moving a product from the cart into a plastic bag hanging at the back of the cart.",
        "The static security camera captures a side-view, full-body shot of a man with a shopping cart in a store aisle transferring a product from the cart into the plastic bag held with his left hand.",
        "A full-body, side-facing shot from a static surveillance camera shows a man with a shopping cart in an aisle moving a product from the cart into the plastic bag hung from his left elbow in front of his chest.",
        "From a static surveillance camera's side-view, full-body perspective, a man with a shopping cart in an aisle moves a product from the cart into the bag slung over his left shoulder.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He slides the product into his coat pocket.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He lowers the product to waist hip, and gradually tucks the product into his pants pocket.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He carefully and slowly slides the product into his back pocket.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He places the product into the inner-left pocket of his coat.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He lowers the product into his tote at hip level.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He places the product  into his shoulder bag at chest level.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He places the product  into his crossbody bag.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He places the product into his right shoulder bag with his right hand, while keeping his left hand steady.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He puts the product into a plastic bag hanging at the back of the cart.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He slips the product into the personal plastic bag held with his left hand.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He slips the product into the personal plastic bag hung from his left elbow in front of his chest.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He places the product into the bag slinging over his left shoulder.",
        "A **fixed surveillance camera** records a **man** with a **shopping cart** in an **aisle** from a **full-body, side-view** as he **moves a product from the cart into a plastic bag hanging at the back of the cart**.",
        "The **static security camera** captures a **full-body, side-view shot of a man** in a **store aisle** with a **shopping cart** as he **transfers a product from the cart into the plastic bag held with his left hand**.",
        "A **man** with a **shopping cart** in an **aisle** is shown in a **full-body, side-facing shot from a static surveillance camera** **moving a product from the cart into the plastic bag hung from his left elbow in front of his chest**.",
        "From a **static surveillance camera's side-view, full-body perspective**, a **man** with a **shopping cart** in an **aisle** **moves a product from the cart into the bag slung over his left shoulder**.",
        "A **man** in a **store aisle** is seen in a **front-view, full-body shot captured by a static surveillance camera**. He **slides the product into his coat pocket**.",
        "A **static surveillance camera** captures a **front-view, full-body view of a man** in a **store aisle**. He **lowers the product to waist hip, and gradually tucks the product into his pants pocket**.",
        "Captured by a **static surveillance camera** in a **front-view, full-body shot**, a **man** in a **store aisle** **carefully and slowly slides the product into his back pocket**.",
        "In a **front-view, full-body shot** from a **static surveillance camera**, a **man** in a **store aisle** **places the product into the inner-left pocket of his coat**.",
        "A **man** in a **store aisle** **lowers the product into his tote at hip level**, captured in a **front-view, full-body view** by a **static surveillance camera**.",
        "A **static surveillance camera** captures a **front-view, full-body view of a man** in a **store aisle**. He **places the product into his shoulder bag at chest level**.",
        "Captured in a **front-view, full-body view** by a **static surveillance camera**, a **man** in a **store aisle** **places the product into his crossbody bag**.",
        "A **man** in a **store aisle** is seen in a **front-view, full-body view** from a **static surveillance camera**. He **places the product into his right shoulder bag with his right hand, while keeping his left hand steady**.",
        "A **static surveillance camera** captures a **front-view, full-body view of a man** in a **store aisle**. He **puts the product into a plastic bag hanging at the back of the cart**.",
        "In a **front-view, full-body view** captured by a **static surveillance camera**, a **man** in a **store aisle** **slips the product into the personal plastic bag held with his left hand**.",
        "A **man** in a **store aisle** **slips the product into the personal plastic bag hung from his left elbow in front of his chest**, captured in a **front-view, full-body view** by a **static surveillance camera**.",
        "A **static surveillance camera** captures a **front-view, full-body view of a man** in a **store aisle**. He **places the product into the bag slinging over his left shoulder**.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He unwraps a cosmetics item to take out the product insides using both hands at hip height, and slides the product into his coat pocket.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He unwraps a cosmetics item to take out the product insides using both hands at hip height, and lowers the product to waist hip, and gradually tucks the product into his pants pocket.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He unwraps a cosmetics item to take out the product insides using both hands at hip height, and carefully and slowly slides the product into his back pocket.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He unwraps a cosmetics item to take out the product insides using both hands at hip height, and places the product into the inner-left pocket of his coat.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He unwraps a cosmetics item to take out the product insides using both hands at hip height, and lowers the product into his tote at hip level.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He unwraps a cosmetics item to take out the product insides using both hands at hip height, and places the product  into his shoulder bag at chest level.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He unwraps a cosmetics item to take out the product insides using both hands at hip height, and places the product  into his crossbody bag.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He unwraps a cosmetics item to take out the product insides using both hands at hip height, and places the product into his right shoulder bag with his right hand, while keeping his left hand steady.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He unwraps a cosmetics item to take out the product insides using both hands at hip height, and puts the product into a plastic bag hanging at the back of the cart.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He unwraps a cosmetics item to take out the product insides using both hands at hip height, and slips the product into the personal plastic bag held with his left hand.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He unwraps a cosmetics item to take out the product insides using both hands at hip height, and slips the product into the personal plastic bag hung from his left elbow in front of his chest.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He unwraps a cosmetics item to take out the product insides using both hands at hip height, and places the product into the bag slinging over his left shoulder.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He inspects a product on his right hand, and slips the product into the personal plastic bag held with his left hand.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He inspects a product on his right hand, and slips the product into the personal plastic bag hung from his left elbow in front of his chest.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He inspects a product on his right hand, and places the product into the bag slinging over his left shoulder.",
        "A surveillance camera captures a full-body, front-view image of a man in a store aisle. He uses both hands, held at hip level, to unwrap a cosmetic item, remove its contents, and slide the product into his coat pocket.",
        "A surveillance camera records a full-body, front-view of a man in a store aisle. He opens a cosmetic product with both hands at hip height, removes the contents, and then discreetly tucks the product into his pants pocket.",
        "A man in a store aisle, viewed from the front and full-body by a static camera, unwraps a cosmetic item using both hands at hip level, extracts the product, and slowly places it into his back pocket.",
        "A static surveillance camera captures a front-view, full-body shot of a man in an aisle. He uses both hands at hip height to take the product out of its cosmetics packaging and then places the product into the inner-left pocket of his coat.",
        "Captured from the front by a surveillance camera, a man in a store aisle uses both hands at hip level to remove the contents from a cosmetic item's packaging before lowering the product into a tote bag at his hip.",
        "A man is filmed full-body and front-view in a store aisle by a static camera. He unwraps a cosmetic product using both hands at hip height to retrieve the product, which he then puts into his shoulder bag at chest level.",
        "A full-body, front-view surveillance image shows a man in a store aisle unwrapping a cosmetic item with both hands at hip height to take out the product, which he then places into his crossbody bag.",
        "A man in a store aisle, seen from the front by a static surveillance camera, uses both hands at hip height to open a cosmetic item and take out the product; he then places the product into his right shoulder bag with his right hand, keeping his left hand steady.",
        "A surveillance camera records a man full-body and front-view in a store aisle. He unwraps a cosmetic product using both hands at hip height, removes the contents, and puts the product into a plastic bag hanging from the back of the cart.",
        "In a full-body, front-view shot from a static camera, a man in a store aisle unwraps a cosmetic item with both hands at hip level, takes out the product, and slips it into a personal plastic bag held in his left hand.",
        "A static surveillance camera captures a front-view, full-body shot of a man in an aisle. He uses both hands at hip height to unwrap a cosmetic product and extract its contents, which he then slips into a personal plastic bag hanging from his left elbow in front of his chest.",
        "A man is seen full-body and front-view in a store aisle via surveillance camera. He unwraps a cosmetic item using both hands at hip level to take out the product, which he then places into a bag slung over his left shoulder.",
        "A static surveillance camera captures a full-body, front-view of a man in an aisle. He examines a product in his right hand before slipping it into a personal plastic bag held with his left hand.",
        "A man in a store aisle is captured full-body and front-view by a static camera. He inspects a product in his right hand and then slips it into a personal plastic bag hung from his left elbow in front of his chest.",
        "A full-body, front-view surveillance image shows a man in a store aisle inspecting a product in his right hand before placing it into a bag slung over his left shoulder.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He transfers a product from shelf into his coat pocket.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He transfers a product from shelf into his pants pocket.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He transfers a product from shelf into his back pocket.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He transfers a product from shelf into the inner-left pocket of his coat.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He transfers a product from shelf into his tote at hip level.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He transfers a product from shelf into his shoulder bag at chest level.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He transfers a product from shelf into his crossbody bag.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He transfers a product from shelf into his right shoulder bag with his right hand, while keeping his left hand steady.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He transfers a product from shelf into a plastic bag hanging at the back of the cart.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He transfers a product from shelf into the personal plastic bag held with his left hand.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He transfers a product from shelf into the personal plastic bag hung from his left elbow in front of his chest.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He transfers a product from shelf into the bag slinging over his left shoulder.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He transfers a product from the basket into his coat pocket.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He transfers a product from the basket into his pants pocket.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He transfers a product from the basket into his back pocket.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He transfers a product from the basket into the inner-left pocket of his coat.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He transfers a product from the basket into his tote at hip level.",
        "A full-body, front-view security camera recording shows a man in a store aisle putting an item from the shelf into his coat pocket.",
        "A static surveillance camera provides a full-body, front-on view of a man in a store aisle taking a product from the shelf and placing it in his pants pocket.",
        "Footage from a stationary surveillance camera captures a man, full-body and front-facing, in a store aisle as he moves an item from the shelf into his back pocket.",
        "A front-view, full-body shot from a static security camera in a store aisle shows a man concealing a product from the shelf in his coat's inner-left pocket.",
        "In a store aisle, a man is seen in full-body, front-view surveillance footage moving a product from the shelf into his tote bag carried at hip level.",
        "A static surveillance camera captures a full-body, front-view of a man in a store aisle placing a product from the shelf into his shoulder bag, which is positioned at chest level.",
        "A man in a store aisle transfers an item from the shelf into his crossbody bag, as recorded by a full-body, front-view static surveillance camera.",
        "A full-body, front-view static security camera shows a man in a store aisle putting a product from the shelf into his right shoulder bag using his right hand, while his left hand remains still.",
        "Surveillance footage shows a man, full-body and front-view, in a store aisle transferring a product from the shelf into a plastic bag hanging on the back of the shopping cart.",
        "A man in a store aisle places a product from the shelf into the personal plastic bag he holds with his left hand, as captured by a full-body, front-view static surveillance camera.",
        "A static surveillance camera captures a front-view, full-body image of a man in a store aisle moving a product from the shelf into a personal plastic bag looped over his left elbow in front of his chest.",
        "A full-body, front-view static surveillance camera shows a man in a store aisle taking a product from the shelf and putting it into the bag slung over his left shoulder.",
        "A static surveillance camera provides a front-view, full-body perspective of a man in a store aisle putting a product from the shopping basket into his coat pocket.",
        "Footage from a stationary surveillance camera captures a full-body, front-view of a man in a store aisle moving a product from the shopping basket into his pants pocket.",
        "A man in a store aisle transfers an item from the shopping basket into his back pocket, recorded by a static surveillance camera in a full-body, front-view shot.",
        "A full-body, front-view static security camera captures a man in a store aisle moving a product from the shopping basket into the inner-left pocket of his coat.",
        "A static surveillance camera shows a man, full-body and front-facing, in a store aisle putting a product from the shopping basket into his tote bag carried at hip level.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He transfers a product from the basket into his shoulder bag at chest level.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He transfers a product from the basket into his crossbody bag.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He transfers a product from the basket into his right shoulder bag with his right hand, while keeping his left hand steady.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He transfers a product from the basket into a plastic bag hanging at the back of the cart.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He transfers a product from the basket into the personal plastic bag held with his left hand.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He transfers a product from the basket into the personal plastic bag hung from his left elbow in front of his chest.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He transfers a product from the basket into the bag slinging over his left shoulder.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He transfers a product from the basket held with his left hand into his coat pocket.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He transfers a product from the basket held with his left hand into his pants pocket.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He transfers a product from the basket held with his left hand into his back pocket.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He transfers a product from the basket held with his left hand into the inner-left pocket of his coat.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He transfers a product from the basket held with his left hand into his tote at hip level.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He transfers a product from the basket held with his left hand into his shoulder bag at chest level.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He transfers a product from the basket held with his left hand into his crossbody bag.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He transfers a product from the basket held with his left hand into his right shoulder bag with his right hand, while keeping his left hand steady.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He transfers a product from the basket held with his left hand into a plastic bag hanging at the back of the cart.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He transfers a product from the basket held with his left hand into the personal plastic bag held with his left hand.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He transfers a product from the basket held with his left hand into the personal plastic bag hung from his left elbow in front of his chest.",
        "A static surveillance camera captures a front-view, full-body view of a man in a store aisle. He transfers a product from the basket held with his left hand into the bag slinging over his left shoulder.",
        "A full-body, front-view security camera recording in a store aisle shows a man moving an item from his basket into a shoulder bag near his chest.",
        "A man in a store aisle is seen on full-body, front-view surveillance footage transferring a product from his basket into a crossbody bag.",
        "Surveillance footage captures a man in a store aisle moving an item from the basket into his right shoulder bag with his right hand, while his left hand holds the basket steady. The view is a front, full-body shot.",
        "A static, full-body camera angle captures a man in a store aisle putting an item from his basket into a plastic bag hung on the back of his shopping cart.",
        "In a store aisle, a man is captured on front-view, full-body surveillance putting a product from his basket into a personal plastic bag held in his left hand.",
        "A static camera provides a full-body, front view of a man in a store aisle moving an item from his basket into a personal plastic bag hanging from his left elbow in front of his chest.",
        "A man in a store aisle is shown on full-body, front-view security footage placing a product from his basket into a bag slung over his left shoulder.",
        "Full-body, front-view surveillance video shows a man in a store aisle transferring an item from the basket he holds with his left hand into his coat pocket.",
        "A static camera captures a man in a store aisle, full-body and front-view, moving a product from the basket (held in his left hand) into his pants pocket.",
        "In a store aisle, a man is seen on full-body, front-view surveillance putting an item from the basket held with his left hand into his back pocket.",
        "A man in a store aisle is recorded by a full-body, front-view surveillance camera transferring a product from the basket (held with his left hand) into the inner-left pocket of his coat.",
        "A full-body, front-view security camera shows a man in a store aisle placing an item from the basket he holds with his left hand into his tote bag at hip level.",
        "Surveillance footage shows a man in a store aisle moving an item from the basket (held with his left hand) into his shoulder bag at chest level. The camera angle is full-body and front-view.",
        "A man in a store aisle is captured on full-body, front-view surveillance moving a product from the basket (held with his left hand) into his crossbody bag.",
        "A static, full-body, front-view camera records a man in a store aisle transferring an item from the basket (held with his left hand) into his right shoulder bag using his right hand, keeping his left hand steady.",
        "In a store aisle, a man is seen on full-body, front-view surveillance taking a product from the basket (held with his left hand) and putting it into a plastic bag hanging on the back of the cart.",
        "A full-body, front-view security camera records a man in a store aisle transferring a product from the basket he holds with his left hand into a personal plastic bag also held with his left hand.",
        "A man in a store aisle, captured on full-body, front-view surveillance, moves an item from the basket (held with his left hand) into a personal plastic bag hung from his left elbow in front of his chest.",
        "Full-body, front-view surveillance shows a man in a store aisle placing an item from the basket (held with his left hand) into the bag slung over his left shoulder.",
        "A static surveillance camera captures a front-view, full-body view of a man with a shopping cart in a store aisle. He transfers a product from the shopping cart into his coat pocket.",
        "A static surveillance camera captures a front-view, full-body view of a man with a shopping cart in a store aisle. He transfers a product from the shopping cart into his pants pocket.",
        "A static surveillance camera captures a front-view, full-body view of a man with a shopping cart in a store aisle. He transfers a product from the shopping cart into his back pocket.",
        "A static surveillance camera captures a front-view, full-body view of a man with a shopping cart in a store aisle. He transfers a product from the shopping cart into the inner-left pocket of his coat.",
        "A static surveillance camera captures a front-view, full-body view of a man with a shopping cart in a store aisle. He transfers a product from the shopping cart into his tote at hip level.",
        "A static surveillance camera captures a front-view, full-body view of a man with a shopping cart in a store aisle. He transfers a product from the shopping cart into his shoulder bag at chest level.",
        "A static surveillance camera captures a front-view, full-body view of a man with a shopping cart in a store aisle. He transfers a product from the shopping cart into his crossbody bag.",
        "A static surveillance camera captures a front-view, full-body view of a man with a shopping cart in a store aisle. He transfers a product from the shopping cart into his right shoulder bag with his right hand, while keeping his left hand steady.",
        "A static surveillance camera captures a front-view, full-body view of a man with a shopping cart in a store aisle. He transfers a product from the shopping cart into a plastic bag hanging at the back of the cart.",
        "A static surveillance camera captures a front-view, full-body view of a man with a shopping cart in a store aisle. He transfers a product from the shopping cart into the personal plastic bag held with his left hand.",
        "A static surveillance camera captures a front-view, full-body view of a man with a shopping cart in a store aisle. He transfers a product from the shopping cart into the personal plastic bag hung from his left elbow in front of his chest.",
        "A static surveillance camera captures a front-view, full-body view of a man with a shopping cart in a store aisle. He transfers a product from the shopping cart into the bag slinging over his left shoulder.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He slides the product into his coat pocket.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He lowers the product to waist hip, and gradually tucks the product into his pants pocket.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He carefully and slowly slides the product into his back pocket.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He places the product into the inner-left pocket of his coat.",
        "A static surveillance camera captures a side-view, full-body view of a man in a store aisle. He lowers the product into his tote at hip level.",
        "A store's static surveillance camera records a full-body, front-facing image of a man pushing a shopping cart in an aisle. He is seen moving an item from the cart into his coat pocket.",
        "A security camera in a store captures a front view of a man with a shopping cart in an aisle, showing his whole body. He puts an item from the cart into his pants pocket.",
        "The full-body, front-view footage from a static surveillance camera in a store aisle shows a man with a cart. He transfers a product from the cart to his back pocket.",
        "In a store aisle, a static surveillance camera provides a full, front-body view of a man using a shopping cart. He places a product from the cart into the inner-left pocket of his coat.",
        "A full-body, front-view shot from a static surveillance camera in a store aisle captures a man with a shopping cart. He moves a product from the cart into his hip-level tote bag.",
        "The static surveillance footage from a store aisle shows a man, full-body and front-view, with a shopping cart. He places a product from the cart into his shoulder bag, which is at chest level.",
        "A full-body, front-view image of a man with a shopping cart in a store aisle is captured by a static camera. He transfers a product from the cart into his crossbody bag.",
        "A man in a store aisle, seen full-body and front-view by a static camera, is pushing a shopping cart. Using his right hand, he moves a product from the cart into his right shoulder bag, keeping his left hand still.",
        "A static surveillance camera captures a full-body, front view of a man and his shopping cart in an aisle. He places an item from the cart into a plastic bag hanging on the back of the cart.",
        "A full-body, front-view recording by a static surveillance camera shows a man with a shopping cart in a store aisle. He puts a product from the cart into a personal plastic bag he is holding with his left hand.",
        "In a store aisle, a static camera captures a full-body, front view of a man with a shopping cart. He moves a product from the cart into a personal plastic bag that is hanging from his left elbow in front of his chest.",
        "A static surveillance camera captures a front-view, full-body shot of a man with a shopping cart in a store aisle. He transfers an item from the cart into the bag slung over his left shoulder.",
        "A static security camera captures a full-body, side-view of a man in a store aisle. He slides an item into his coat pocket.",
        "The full-body, side-view footage from a static surveillance camera shows a man in a store aisle. He lowers an item to hip level and gradually tucks it into his pants pocket.",
        "A static surveillance camera records a side-view, full-body image of a man in a store aisle. He slowly and carefully slides an item into his back pocket.",
        "The static surveillance footage provides a side-view, full-body shot of a man in a store aisle. He puts an item into the inner-left pocket of his coat.",
        "A store aisle scene is recorded by a static surveillance camera, providing a side-view, full-body shot of a man. He lowers an item into his hip-level tote bag.",
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
        "A surveillance camera provides a full, side-view of a man in a store aisle as he puts a product into his shoulder bag near his chest.",
        "A static, side-view camera records a man in an aisle placing an item into his crossbody bag.",
        "The store's surveillance footage shows a man from the side, fully visible in an aisle, using his right hand to put a product into his right shoulder bag while keeping his left hand steady.",
        "A static, full-body side-view from a surveillance camera shows a man in a store aisle putting a product into a plastic bag hanging on the back of his shopping cart.",
        "A man in a store aisle is captured by a side-view camera slipping an item into a personal plastic bag held in his left hand.",
        "Surveillance footage captures a man, full-body side-view, in an aisle, slipping a product into a personal plastic bag hanging from his left elbow in front of his chest.",
        "A full-body side-view of a man in a store aisle is captured by a static camera as he places a product into a bag slung over his left shoulder.",
        "A man in a store aisle is recorded by a static side-view camera. Using both hands at hip height, he unwraps a cosmetics item to take out the contents and then slides the product into his coat pocket.",
        "A full-body side-view from a surveillance camera shows a man in an aisle unwrapping a cosmetics item using both hands at hip height, taking out the contents, lowering it to his waist, and then gradually tucking the product into his pants pocket.",
        "A static camera captures a full-body side-view of a man in a store aisle. He unwraps a cosmetics item at hip height to get the product inside, and then carefully and slowly slides it into his back pocket.",
        "In a store aisle, a man is filmed by a static, side-view camera. He unwraps a cosmetics item using both hands at hip height to remove the contents, and then places the product into the inner-left pocket of his coat.",
        "A surveillance camera provides a full, side-view of a man in a store aisle who unwraps a cosmetics item at hip height to take out the product, and then lowers the product into his tote bag at hip level.",
        "Captured by a static, full-body side-view camera, a man in an aisle unwraps a cosmetics item using both hands at hip height to take out the contents, and then places the product into his shoulder bag near his chest.",
        "A static, side-view camera captures a full view of a man in a store aisle. He unwraps a cosmetics item at hip height with both hands to get the contents, and then puts the product into his crossbody bag.",
        "A full-body side-view of a man in a store aisle is recorded by a static camera. He uses both hands at hip height to unwrap a cosmetics item and take out the product, and then places it into his right shoulder bag with his right hand while stabilizing it with his left.",
        "The static, side-view camera in a store aisle shows a full-body view of a man. He unwraps a cosmetics item using both hands at hip height, takes out the product, and then puts it into a plastic bag hanging on the back of the cart.",
        "A man in a store aisle is seen in a full-body side-view by a static camera. After unwrapping a cosmetics item at hip height to remove the product, he slips the item into a personal plastic bag held in his left hand.",
        "In a store aisle, a man is filmed by a static, side-view camera. He uses both hands at hip height to unwrap a cosmetics item and take out the product, and then slips the product into a personal plastic bag hung from his left elbow in front of his chest.",
        "A static camera captures a full-body side-view of a man in a store aisle. He unwraps a cosmetics item at hip height to take out the product, and then places the product into the bag slung over his left shoulder.",
        "A static surveillance camera captures a full-body, side-view of a man in a store aisle. He inspects a product in his right hand before slipping it into a personal plastic bag held with his left hand.",
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
        "A full-body, side-view surveillance video shows a man in a store aisle examining an item with his right hand before slipping it into a plastic bag hanging from his left elbow in front of his chest.",
        "Footage from a stationary side-view camera in a store aisle captures a man inspecting a product in his right hand and then placing it into a bag hanging over his left shoulder.",
        "A man in a store aisle is seen on a static side-view camera transferring an item from the shelf directly into his coat pocket.",
        "A stationary surveillance camera records a man in a store aisle moving a product from the shelf into his pants pocket.",
        "A full-body, side-view shot shows a man in a store aisle taking an item from the shelf and putting it into his back pocket.",
        "A static surveillance camera captures a man in a store aisle moving a product from the shelf into the inner-left pocket of his coat.",
        "A side-view, full-body surveillance image shows a man taking a product from the shelf and placing it into his tote bag at hip level.",
        "A man in a store aisle is captured on a static side-view camera moving a product from the shelf into his shoulder bag, which is positioned at chest level.",
        "Footage from a stationary surveillance camera shows a man in a store aisle transferring a product from the shelf into his crossbody bag.",
        "A man in a store aisle is recorded transferring an item from the shelf into his right shoulder bag with his right hand, keeping his left hand still, as seen by a static side-view camera.",
        "A static surveillance camera captures a full-body, side-view of a man in an aisle moving a product from the shelf into a plastic bag secured to the back of his shopping cart.",
        "A man in a store aisle is seen on a stationary side-view camera transferring a product from the shelf into a personal plastic bag held in his left hand.",
        "A side-view, full-body surveillance image captures a man in a store aisle transferring a product from the shelf into a personal plastic bag hanging from his left elbow in front of his chest.",
        "A static side-view camera shows a man in a store aisle moving a product from the shelf into a bag slung over his left shoulder.",
        "A man in a store aisle is captured on a static, side-view camera transferring an item from his shopping basket into his coat pocket.",
        "A stationary surveillance camera records a man in a store aisle moving a product from his basket into his pants pocket.",
        "A full-body, side-view shot captures a man in a store aisle taking a product from his basket and putting it into his back pocket.",
        "A man in a store aisle is seen on a static side-view camera transferring a product from the basket into the inner-left pocket of his coat.",
        "A side-view, full-body surveillance image shows a man taking a product from the basket and placing it into his tote at hip level.",
        "Footage from a stationary side-view camera captures a man in a store aisle moving a product from his basket into his shoulder bag, which is at chest level.",
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
        "A surveillance camera provides a side-view, full-body shot of a man in a store aisle as he moves an item from his basket to his crossbody bag.",
        "The footage shows a man standing in a store aisle, captured from the side, moving a product from his basket into his right shoulder bag using his right hand, while his left hand remains still.",
        "A static surveillance camera records a man in a store aisle, viewed from the side, placing an item from his basket into a plastic bag hanging on the back of his shopping cart.",
        "A side-view, full-body shot from a static camera shows a man in a store aisle putting an item from his basket into a plastic bag he holds in his left hand.",
        "In a store aisle, a static camera captures a man from the side as he transfers a product from his basket into a plastic bag that is suspended from his left elbow near his chest.",
        "A surveillance camera captures a side-view of a man in an aisle, showing him moving an item from his basket into the bag slung over his left shoulder.",
        "The camera footage reveals a man in a store aisle, seen from the side, putting an item from the basket he holds in his left hand into his coat pocket.",
        "A side-view, full-body shot of a man in a store aisle shows him transferring an item from his left-hand held basket into his pants pocket.",
        "A man in a store aisle is seen from the side by a static camera, taking an item from the basket in his left hand and putting it into his back pocket.",
        "A surveillance camera captures a side-view of a man in an aisle, transferring a product from the basket held in his left hand into the inner-left pocket of his coat.",
        "The footage shows a man in a store aisle, viewed from the side, moving an item from his left-hand held basket into his tote bag at hip level.",
        "A static camera in a store aisle captures a man from the side, placing an item from the basket in his left hand into his shoulder bag at chest level.",
        "A man in a store aisle, captured in a side-view, full-body shot, moves a product from the basket he holds in his left hand into his crossbody bag.",
        "A man is seen in a store aisle, viewed from the side by a static camera, transferring a product from the basket in his left hand to his right shoulder bag using his right hand, while his left hand is steady.",
        "The static camera footage shows a man in a store aisle, from the side, moving an item from the basket he holds in his left hand into a plastic bag hanging from the back of the cart.",
        "A side-view camera captures a man in an aisle transferring a product from the basket, held in his left hand, into a personal plastic bag, also held in his left hand.",
        "The scene shows a man in a store aisle, captured from the side, placing a product from the basket in his left hand into a personal plastic bag hung from his left elbow in front of his chest.",
        "A static surveillance camera captures a man in a store aisle, seen from the side, moving an item from the basket held in his left hand into the bag slinging over his left shoulder.",
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
        "A surveillance camera provides a full-body, side-view of a man pushing a shopping cart in a store aisle as he moves an item from the cart to his coat pocket.",
        "The footage from a fixed surveillance camera shows a man, seen in a full-body side-view, in a store aisle with a cart; he places a product from the cart into his pants pocket.",
        "A static, side-view, full-body shot from a security camera in a store aisle shows a man with a shopping cart concealing an item from the cart in his back pocket.",
        "In a store aisle, a man with a shopping cart is captured by a side-view security camera as he transfers a product from the cart to the inner-left pocket of his coat.",
        "A static surveillance camera captures a man in a store aisle from the side, showing his full body, as he takes an item from his shopping cart and puts it into his tote bag at hip level.",
        "Footage from a fixed surveillance camera in a store aisle shows a full-body side-view of a man with a shopping cart putting a product from the cart into his shoulder bag, which is at chest level.",
        "A full-body side-view of a man with a shopping cart in a store aisle is captured by a security camera as he transfers a product from the cart into his crossbody bag.",
        "A static surveillance camera records a full-body side-view of a man in a store aisle with a shopping cart, showing him using his right hand to move a product from the cart into his right shoulder bag while his left hand remains steady.",
        "In a store aisle, a man with a shopping cart is seen in a full-body side-view by a static camera as he places a product from the cart into a plastic bag hanging on the back of the cart.",
        "A static surveillance camera in a store aisle captures a full-body side-view of a man with a shopping cart moving an item from the cart into a personal plastic bag held with his left hand.",
        "A security camera in a store aisle shows a man in a full-body side-view with a shopping cart, as he transfers a product from the cart into his personal plastic bag, which is hanging from his left elbow in front of his chest.",
        "A static surveillance camera provides a full-body side-view of a man with a shopping cart in a store aisle as he places a product from the cart into a bag slung over his left shoulder.",
    ]
    args.output_path = "/home/lap_awlv/CogVideo/outputs/cogvideo_16/videos"
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
