export CUDA_VISIBLE_DEVICES=0

model_name=VLBM

python -u run_vlbm.py \
  --task_name ood_vlbm \
  --is_training 1 \
  --root_path ./raw_files/PSM/ \
  --model_id psm_96_12 \
  --model $model_name \
  --data csv_by_ood_vlbm \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 12 \
  --hiddens 128 \
  --num_Period_Block 1 \
  --m 50 \
  --interval 1 \
  --heads 1 \
  --learning_rate 0.001 \
  --enc_in 25 \
  --dec_in 25 \
  --c_out 25 \
  --des 'Exp' \
  --use_norm 0 \
  --batch_size 128 \
  --itr 1

python -u run_vlbm.py \
  --task_name ood_vlbm \
  --is_training 1 \
  --root_path ./raw_files/PSM/ \
  --model_id psm_96_96 \
  --model $model_name \
  --data csv_by_ood_vlbm \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --hiddens 128 \
  --num_Period_Block 1 \
  --m 50 \
  --interval 1 \
  --heads 1 \
  --learning_rate 0.001 \
  --enc_in 25 \
  --dec_in 25 \
  --c_out 25 \
  --des 'Exp' \
  --use_norm 1 \
  --batch_size 128 \
  --itr 1

python -u run_vlbm.py \
  --task_name ood_vlbm \
  --is_training 1 \
  --root_path ./raw_files/PSM/ \
  --model_id psm_96_192 \
  --model $model_name \
  --data csv_by_ood_vlbm \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 192 \
  --hiddens 256 \
  --num_Period_Block 1 \
  --m 50 \
  --interval 1 \
  --heads 4 \
  --learning_rate 0.001 \
  --enc_in 25 \
  --dec_in 25 \
  --c_out 25 \
  --des 'Exp' \
  --use_norm 1 \
  --batch_size 128 \
  --itr 1

python -u run_vlbm.py \
  --task_name ood_vlbm \
  --is_training 1 \
  --root_path ./raw_files/PSM/ \
  --model_id psm_96_336 \
  --model $model_name \
  --data csv_by_ood_vlbm \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 336 \
  --hiddens 128 \
  --num_Period_Block 1 \
  --m 50 \
  --interval 1 \
  --heads 1 \
  --learning_rate 0.001 \
  --enc_in 25 \
  --dec_in 25 \
  --c_out 25 \
  --des 'Exp' \
  --use_norm 1 \
  --batch_size 128 \
  --itr 1