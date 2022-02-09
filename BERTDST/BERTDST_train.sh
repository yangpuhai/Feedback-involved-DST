# dataset can be  "WOZ_2.0" or "MultiWOZ_2.1"
dataset="MultiWOZ_2.1"

random_seed=40
n_epochs=200
max_seq_length=150
train_feedback=0.2

nohup python -u BERTDST_train.py \
  --dataset=${dataset} \
  --max_seq_length=${max_seq_length} \
  --batch_size=16 \
  --enc_lr=4e-5 \
  --dec_lr=1e-4 \
  --n_epochs=${n_epochs} \
  --patience=6 \
  --dropout=0.1 \
  --word_dropout=0.1 \
  --train_feedback=${train_feedback} \
  --random_seed=${random_seed} \
  > BERTDST_${dataset}_epoch[${n_epochs}]_maxlen[${max_seq_length}]_feedback[${train_feedback}]_seed[${random_seed}]_train.log 2>&1 &