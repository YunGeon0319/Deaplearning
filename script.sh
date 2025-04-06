for seed in 0 1 2; do
  python 3.31.tlftmq.py \
    --node_num 100 \
    --seed $seed \
    --batch_size 32 \
    --lr 0.01 \
    --epochs 20 \
    --dropout 0.5 \
    --csv_path C:/Users/user/wetie/3.31/log.csv
done
