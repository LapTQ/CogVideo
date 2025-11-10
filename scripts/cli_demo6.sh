
export HF_HOME="/media/home4/free_space/lap_awlv/cache/huggingface"

PATHD_OUTPUT=/home/lap_awlv/CogVideo/outputs/cogvideo_6
mkdir -p $PATHD_OUTPUT

CUDA_VISIBLE_DEVICES=2 python3 inference/cli_demo_6.py \
    --prompt None \
    --generate_type t2v \
    --num_inference_steps 50 \
    --num_videos_per_prompt 5 \
    --output_path $PATHD_OUTPUT/prompt-{}/gen-video-{}.mp4 \
    --model_path THUDM/CogVideoX-5b \
    --num_frames 49 \
    --fps 8 \
    --dtype bfloat16 \
    
    
    
    # --model_path THUDM/CogVideoX-2b \
    # --num_frames 49 \
    # --fps 8 \
    # --dtype float16 \



    # --model_path THUDM/CogVideoX1.5-5B \
    # --num_frames 81 \
    # --fps 16 \
    # --dtype bfloat16 \


    # --image_or_video_path /home/lap_awlv/CogVideo/outputs/gen-images/gen11.jpg \