export CUDA_VISIBLE_DEVICES=0

model_name=VLBM

python -u run_vlbm.py  \
  --task_name long_term_forecast_vlbm \
  --is_training 1 \
  --root_path ./raw_files/weather/ \
  --data_path weather.csv \
  --model_id weather_96_12 \
  --model $model_name \
  --data vlbm \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --hiddens 128 \
  --num_Period_Block 1 \
  --m 100 \
  --interval 10 \
  --heads 1 \
  --learning_rate 0.001 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --use_norm 1 \
  --batch_size 16 \
  --itr 1

python -u run_vlbm.py  \
  --task_name long_term_forecast_vlbm \
  --is_training 1 \
  --root_path ./raw_files/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data vlbm \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 12 \
  --hiddens 128 \
  --num_Period_Block 1 \
  --m 10 \
  --interval 10 \
  --heads 1 \
  --learning_rate 0.001 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --use_norm 0 \
  --batch_size 128 \
  --itr 1

python -u run_vlbm.py  \
  --task_name long_term_forecast_vlbm \
  --is_training 1 \
  --root_path ./raw_files/weather/ \
  --data_path weather.csv \
  --model_id weather_96_336 \
  --model $model_name \
  --data vlbm \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 336 \
  --hiddens 128 \
  --num_Period_Block 1 \
  --m 100 \
  --interval 10 \
  --heads 1 \
  --learning_rate 0.001 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --use_norm 1 \
  --batch_size 16 \
  --itr 1

python -u run_vlbm.py  \
  --task_name long_term_forecast_vlbm \
  --is_training 1 \
  --root_path ./raw_files/weather/ \
  --data_path weather.csv \
  --model_id weather_96_720 \
  --model $model_name \
  --data vlbm \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 720 \
  --hiddens 128 \
  --num_Period_Block 1 \
  --m 50 \
  --interval 10 \
  --heads 1 \
  --learning_rate 0.0003 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --use_norm 1 \
  --batch_size 16 \
  --itr 1