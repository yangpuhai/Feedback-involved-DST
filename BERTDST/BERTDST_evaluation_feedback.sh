# dataset can be  "WOZ_2.0" or "MultiWOZ_2.1"
# feedback_acquisition_strategy can be explicit or implicit
# feedback_content can be belief_state or turn_label
# feedback_timing can be turn, task or session
dataset="MultiWOZ_2.1"
feedback_acquisition_strategy='explicit'
feedback_content='belief_state'
feedback_timing='turn'

random_seed=40
n_epochs=200
max_seq_length=150

train_feedback=1.0

nohup python -u BERTDST_evaluation_feedback.py \
  --dataset=${dataset} \
  --max_seq_length=${max_seq_length} \
  --n_epochs=${n_epochs} \
  --random_seed=${random_seed} \
  --train_feedback=${train_feedback} \
  --feedback_acquisition_strategy=${feedback_acquisition_strategy} \
  --feedback_content=${feedback_content} \
  --feedback_timing=${feedback_timing} \
  > BERTDST_${dataset}_epoch[${n_epochs}]_maxlen[${max_seq_length}]_feedback[${train_feedback}]_seed[${random_seed}]_[${feedback_acquisition_strategy}]_[${feedback_content}]_[${feedback_timing}]_evaluation_feedback.log 2>&1 &