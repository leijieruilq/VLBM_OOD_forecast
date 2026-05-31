export CUDA_VISIBLE_DEVICES=0

model_name=VLBM

python -u run_vlbm.py \
  --task_name long_term_forecast_vlbm \
  --is_training 1 \
  --root_path ./raw_files/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_12 \
  --model $model_name \
  --data vlbm \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 12 \
  --hiddens 256 \
  --num_Period_Block 4 \
  --m 800 \
  --interval 60 \
  --heads 4 \
  --learning_rate 0.001 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --use_norm 1 \
  --batch_size 16 \
  --itr 1

python -u run_vlbm.py \
  --task_name long_term_forecast_vlbm \
  --is_training 1 \
  --root_path ./raw_files/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model $model_name \
  --data vlbm \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --hiddens 256 \
  --num_Period_Block 4 \
  --m 800 \
  --interval 60 \
  --heads 4 \
  --learning_rate 0.001 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --use_norm 1 \
  --batch_size 16 \
  --itr 1

python -u run_vlbm.py \
  --task_name long_term_forecast_vlbm \
  --is_training 1 \
  --root_path ./raw_files/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model $model_name \
  --data vlbm \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 336 \
  --hiddens 256 \
  --num_Period_Block 4 \
  --m 800 \
  --interval 60 \
  --heads 4 \
  --learning_rate 0.001 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --use_norm 1 \
  --batch_size 16 \
  --itr 1

python -u run_vlbm.py \
  --task_name long_term_forecast_vlbm \
  --is_training 1 \
  --root_path ./raw_files/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model $model_name \
  --data vlbm \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 720 \
  --hiddens 256 \
  --num_Period_Block 4 \
  --m 800 \
  --interval 60 \
  --heads 4 \
  --learning_rate 0.001 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --use_norm 1 \
  --batch_size 16 \
  --itr 1
