export MODEL_NAME="/root/autodl-tmp/stable-diffusion-3-medium-diffusers/"
export INSTANCE_DIR="/root/autodl-tmp/datas/5_sd3"
export OUTPUT_DIR="bl_sd3_128"

accelerate launch train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --instance_prompt="a style of black light style" \
  --resolution=1024 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --checkpointing_steps=200\
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="cosine_with_restarts" \
  --lr_num_cycles=4 \
  --lr_warmup_steps=0 \
  --num_train_epochs=200 \
  --validation_width=768 \
  --validation_height=1344 \
  --validation_prompt="black light style, A woman with a huge hoop eardrop, pinned hair clip, oblique 45 degrees looking at the camera, purple glasses, the whole picture is dim, but highlights the purple atmosphere" \
  --validation_epochs=10 \
  --seed="0" \
  --optimizer="AdamW" \
  --weighting_scheme="logit_normal" \
  --rank=128