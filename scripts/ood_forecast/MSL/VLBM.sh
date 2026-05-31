export CUDA_VISIBLE_DEVICES=0

model_name=VLBM

python -u run_vlbm.py \
  --task_name ood_vlbm \
  --is_training 1 \
  --root_path ./raw_files/MSL/ \
  --model_id msl_96_12 \
  --model $model_name \
  --data npy_by_vlbm \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 12 \
  --hiddens 128 \
  --num_Period_Block 1 \
  --m 50 \
  --interval 1 \
  --heads 4 \
  --learning_rate 0.0005 \
  --enc_in 55 \
  --dec_in 55 \
  --c_out 55 \
  --des 'Exp' \
  --use_norm 1 \
  --batch_size 32 \
  --train_epochs 30 \
  --lradj cosine \
  --itr 1

python -u run_vlbm.py \
  --task_name ood_vlbm \
  --is_training 1 \
  --root_path ./raw_files/MSL/ \
  --model_id msl_96_48  \
  --model $model_name \
  --data npy_by_vlbm \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 48 \
  --hiddens 128 \
  --num_Period_Block 2 \
  --m 100 \
  --interval 1 \
  --heads 1 \
  --learning_rate 0.0005 \
  --enc_in 55 \
  --dec_in 55 \
  --c_out 55 \
  --des 'Exp' \
  --use_norm 1 \
  --batch_size 32 \
  --train_epochs 30 \
  --itr 1

python -u run_vlbm.py \
  --task_name ood_vlbm \
  --is_training 1 \
  --root_path ./raw_files/MSL/ \
  --model_id msl_96_96 \
  --model $model_name \
  --data npy_by_vlbm \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --hiddens 128 \
  --num_Period_Block 1 \
  --m 50 \
  --interval 1 \
  --heads 4 \
  --learning_rate 0.0005 \
  --enc_in 55 \
  --dec_in 55 \
  --c_out 55 \
  --des 'Exp' \
  --use_norm 1 \
  --batch_size 32 \
  --train_epochs 30 \
  --lradj cosine \
  --itr 1

python -u run_vlbm.py \
  --task_name ood_vlbm \
  --is_training 1 \
  --root_path ./raw_files/MSL/ \
  --model_id msl_96_192  \
  --model $model_name \
  --data npy_by_vlbm \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 192 \
  --hiddens 256 \
  --num_Period_Block 1 \
  --m 50 \
  --interval 1 \
  --heads 4 \
  --learning_rate 0.0005 \
  --enc_in 55 \
  --dec_in 55 \
  --c_out 55 \
  --des 'Exp' \
  --use_norm 1 \
  --batch_size 32 \
  --train_epochs 30 \
  --lradj cosine \
  --itr 1