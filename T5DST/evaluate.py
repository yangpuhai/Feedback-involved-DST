
import json
# Strict match evaluation from https://github.com/jasonwu0731/trade-dst/blob/master/models/TRADE.py
# check utils/prediction_sample.json for the format of predictions

def compute_acc(gold, pred, slot_temp):
    miss_gold = 0
    miss_slot = []
    for g in gold:
        if g not in pred:
            miss_gold += 1
            miss_slot.append(g.rsplit("-", 1)[0])
    wrong_pred = 0
    for p in pred:
        if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
            wrong_pred += 1
    ACC_TOTAL = len(slot_temp)
    ACC = len(slot_temp) - miss_gold - wrong_pred
    ACC = ACC / float(ACC_TOTAL)
    return ACC

def compute_prf(gold, pred):
    TP, FP, FN = 0, 0, 0
    if len(gold)!= 0:
        count = 1
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1
        for p in pred:
            if p not in gold:
                FP += 1
        precision = TP / float(TP+FP) if (TP+FP)!=0 else 0
        recall = TP / float(TP+FN) if (TP+FN)!=0 else 0
        F1 = 2 * precision * recall / float(precision + recall) if (precision+recall)!=0 else 0
    else:
        if len(pred)==0:
            precision, recall, F1, count = 1, 1, 1, 1
        else:
            precision, recall, F1, count = 0, 0, 0, 1
    return F1, recall, precision, count

# def state_equal(pred_belief, turn_belief, slot_meta, value_dict):
#     pred_dialog_state = {}
#     gold_dialog_state = {}

#     equal = True
#     for slot in slot_meta:
#         pred_value = pred_dialog_state.get(slot)
#         gold_value = gold_dialog_state.get(slot)
#         if pred_value != gold_value:
#             equal = False
#             for s in value_dict:
#                 if pred_value in [s]+value_dict[s]:
#                     for s1 in [s]+value_dict[s]:
#                         if s1 == gold_value:
#                             equal = True
#                             pred_dialog_state[slot] = s
#                             break
#     return pred_dialog_state, equal

def evaluate_metrics(all_prediction, SLOT_LIST):
    total, turn_acc, joint_acc, F1_pred, F1_count = 0, 0, 0, 0, 0
    for idx, dial in all_prediction.items():
        for k, cv in dial["turns"].items():
            if set(cv["turn_belief"]) == set(cv["pred_belief"]):
                joint_acc += 1
            # else:
            #     print('turn_id',idx + ' ' + k)
            #     print(cv["turn_belief"])
            #     print(cv["pred_belief"])
            #     print("==================")
            total += 1

            # Compute prediction slot accuracy
            temp_acc = compute_acc(set(cv["turn_belief"]), set(cv["pred_belief"]), SLOT_LIST)
            turn_acc += temp_acc

            # Compute prediction joint F1 score
            temp_f1, temp_r, temp_p, count = compute_prf(set(cv["turn_belief"]), set(cv["pred_belief"]))
            F1_pred += temp_f1
            F1_count += count

    joint_acc_score = joint_acc / float(total) if total!=0 else 0
    turn_acc_score = turn_acc / float(total) if total!=0 else 0
    F1_score = F1_pred / float(F1_count) if F1_count!=0 else 0
    return joint_acc_score, F1_score, turn_acc_score

