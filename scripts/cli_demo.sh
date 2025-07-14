
export HF_HOME="/mnt/hdd10tb/Users/laptq/cache/huggingface"

CUDA_VISIBLE_DEVICES=1 python3 inference/cli_demo.py \
    --prompt "A man strolls through the brightly lit aisles of a modern supermarket, his sleek shoulder bag slung casually across his body. With a focused gaze, he reaches up to a high shelf, plucking a neatly packaged item. Glancing furtively around, he then swiftly yet carefully tucks the product away into his personal bag, mindful of his surroundings as a very high camera captures the scene from an elevated vantage point several meters away." \
    --model_path THUDM/CogVideoX-5b \
    --num_frames 49 \
    --fps 8 \
    --dtype float16 \
    --generate_type t2v \
    --num_inference_steps 50 \
    --num_videos_per_prompt 1 \
    --output_path /home/laptq/CogVideo/outputs/output.mp4 \
    # --image_or_video_path /home/laptq/CogVideo/outputs/gen-images/gen11.jpg \


    # --model_path THUDM/CogVideoX1.5-5B \
    # --num_frames 81 \
    # --fps 16 \
    # --dtype bfloat16 \
