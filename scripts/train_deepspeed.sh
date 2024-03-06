source .env

echo model_name_or_path: $LLMs_ROOT_DIR/$MODEL_NAME


deepspeed --num_gpus 8 --master_port=9901 src/train_bash.py \
    --deepspeed configs/ds_config.json \
    --stage pt \
    --do_train \
    --model_name_or_path $LLMs_ROOT_DIR/$MODEL_NAME \
    --dataset wiki_demo \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir $OUTPUTS_ROOT_DIR/$MODEL_NAME-pt \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16 \
    --overwrite_output_dir