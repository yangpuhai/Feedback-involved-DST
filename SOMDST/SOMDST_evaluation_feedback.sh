# dataset can be "WOZ_2.0" or "MultiWOZ_2.1"
# feedback_acquisition_strategy can be explicit or implicit
# feedback_content can be belief_state or turn_label
# feedback_timing can be turn, task or session
dataset="MultiWOZ_2.1"

feedback_acquisition_strategy='explicit'
feedback_content='belief_state'
feedback_timing='session'

random_seed=42
train_feedback=1.0

nohup python -u SOMDST_evaluation_feedback.py \
  --dataset=${dataset} \
  --max_seq_length=256 \
  --train_feedback=${train_feedback} \
  --feedback_acquisition_strategy=${feedback_acquisition_strategy} \
  --feedback_content=${feedback_content} \
  --feedback_timing=${feedback_timing} \
  --random_seed=${random_seed} \
  > SOMDST_${dataset}_feedback[${train_feedback}]_${feedback_acquisition_strategy}_${feedback_content}_${feedback_timing}_seed[${random_seed}]_evaluation_feedback.log 2>&1 &