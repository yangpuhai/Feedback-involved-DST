# dataset can be "WOZ_2.0" or "MultiWOZ_2.1"
dataset="MultiWOZ_2.1"

random_seed=40
train_feedback=1.0

nohup python -u TRADE_train.py \
  --dataset=${dataset} \
  --batch_size=32 \
  --lr=1e-3 \
  --n_epochs=200 \
  --patience=6 \
  --dropout=0.1 \
  --word_dropout=0.1 \
  --train_feedback=${train_feedback} \
  --random_seed=${random_seed} \
  --hidden_size=400 \
  > TRADE_${dataset}_feedback[${train_feedback}]_seed[${random_seed}]_train.log 2>&1 &