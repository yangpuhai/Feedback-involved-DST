# dataset can be "WOZ_2.0" or "MultiWOZ_2.1"
dataset="MultiWOZ_2.1"

random_seed=42
train_feedback=1.0

nohup python -u SOMDST_train.py \
  --dataset=${dataset} \
  --max_seq_length=256 \
  --batch_size=32 \
  --enc_lr=4e-5 \
  --dec_lr=1e-4 \
  --n_epochs=30 \
  --patience=6 \
  --dropout=0.1 \
  --word_dropout=0.1 \
  --train_feedback=${train_feedback} \
  --random_seed=${random_seed} \
  > SOMDST_${dataset}_feedback[${train_feedback}]_seed[${random_seed}]_train.log 2>&1 &