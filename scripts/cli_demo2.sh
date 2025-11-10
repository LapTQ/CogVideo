
export HF_HOME="/media/home4/free_space/lap_awlv/cache/huggingface"

PATHD_OUTPUT=/home/lap_awlv/CogVideo/outputs/cogvideo_2
mkdir -p $PATHD_OUTPUT

CUDA_VISIBLE_DEVICES=1 python3 inference/cli_demo.py \
    --prompt "The camera captures a full-body view of a man shopping. He has a shoulder bag slung across him. In one hand, he holds a product, then he quickly but carefully places the product in his bag while observing his surroundings." \
    --generate_type t2v \
    --num_inference_steps 50 \
    --num_videos_per_prompt 20 \
    --output_path $PATHD_OUTPUT/gen-video-{}.mp4 \
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