
import os, random
os.environ['CUDA_VISIBLE_DEVICES']="2"

import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from transformers import (AdamW, T5Tokenizer, BartTokenizer, BartForConditionalGeneration, T5ForConditionalGeneration, WEIGHTS_NAME,CONFIG_NAME)
from data_loader import prepare_data
from evaluate import evaluate_metrics, compute_acc, compute_prf
from copy import deepcopy
import json
from tqdm import tqdm
from T5 import DST_Seq2Seq

import sys
sys.path.append("..")
# print(sys.path)
import info

def test(args, *more):
    args_1 = deepcopy(args)
    args = vars(args)
    args["model_name"] = args["model_checkpoint"]+ "_slotlang_" +str(args["slot_lang"]) + "_lr_" +str(args["lr"]) + "_epoch_" + str(args["n_epochs"]) + "_seed_" + str(args["seed"]) + "_feedback_" + str(args["train_feedback"])
    save_path = os.path.join(args["saving_dir"], args["dataset"], args["model_name"])

    model = T5ForConditionalGeneration.from_pretrained(save_path)
    tokenizer = T5Tokenizer.from_pretrained(save_path)

    task = DST_Seq2Seq(args, tokenizer, model)

    feedback_model = T5ForConditionalGeneration.from_pretrained(args_1.feedback_model_name)
    feedback_tokenizer = T5Tokenizer.from_pretrained(args_1.feedback_model_name)

    data_test, test_loader, ALL_SLOTS = prepare_data(args, task.tokenizer)

    print("test start...")
    #evaluate model
    _ = evaluate_model(args_1, task.tokenizer, task.model, feedback_model, feedback_tokenizer, data_test, save_path, ALL_SLOTS)

def acquire_state(batch_data, model, tokenizer, device):
    input_batch = tokenizer(batch_data["intput_text"], padding=True, return_tensors="pt", add_special_tokens=False, verbose=False)
    batch_data["encoder_input"] = input_batch["input_ids"]
    batch_data["attention_mask"] = input_batch["attention_mask"]
    dst_outputs = model.generate(input_ids=batch_data["encoder_input"].to(device),
                            attention_mask=batch_data["attention_mask"].to(device),
                            eos_token_id=tokenizer.eos_token_id,
                            max_length=200,
                            )
    value_batch = tokenizer.batch_decode(dst_outputs, skip_special_tokens=True)
    pred_state = {s:v for s, v in zip(batch_data["slot_text"], value_batch) if v != "none"}
    gold_state = batch_data["turn_dialog_state"][0]
    equal = False
    if pred_state == gold_state:
        equal = True
    return pred_state, equal

def acquire_corrected_state(args, batch_data, added_information, deleted_information, model, tokenizer, 
    feedback_model, feedback_tokenizer, device):
    feedback_added_information = deepcopy(added_information)
    feedback_deleted_information = deepcopy(deleted_information)
    feedback = info.simulated_negative_feedback(args.dataset, 'test', feedback_added_information, feedback_deleted_information, feedback_model, feedback_tokenizer, device)
    intput_text = [t.replace(batch_data['dialog_history'][i], batch_data['dialog_history'][i] + ' ' + feedback) for i, t in enumerate(batch_data["intput_text"])]
    batch_data["intput_text"] = intput_text
    last_dialog_state_2, equal = acquire_state(batch_data, model, tokenizer, device)
    return last_dialog_state_2, equal

def acquire_pred_gold(args, gold_turn_state, pred_turn_state, last_state, slot_meta):
    if args.feedback_content == 'belief_state':
        pred_state = {s:pred_turn_state.get(s, 'none') for s in slot_meta}
        # print(pred_state)
        gold_state = {s:gold_turn_state.get(s, 'none') for s in slot_meta}
        # print(gold_state)
    elif args.feedback_content == 'turn_label':
        pred_state = {s:pred_turn_state.get(s, 'none') for s in slot_meta if pred_turn_state.get(s, 'none') != last_state.get(s, 'none')}
        # print(pred_state)
        gold_state = {s:gold_turn_state.get(s, 'none') for s in pred_state}
        # print(gold_state)
    else:
        print('select feedback_content in belief_state or turn_label')
        exit()
    return pred_state, gold_state

def evaluate_model(args, tokenizer, model, feedback_model, feedback_tokenizer, data_test, save_path, ALL_SLOTS):
    save_path = os.path.join(save_path,"feedback_results")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    predictions = {}
    # to gpu
    # gpu = args["GPU"][0]
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()

    feedback_model.to(device)
    feedback_model.eval()

    slot_logger = {slot_name:[0,0,0] for slot_name in ALL_SLOTS}

    count = 0
    max_count = len(data_test)
    num_slot = len(ALL_SLOTS)

    slot_turn_acc, joint_acc, slot_F1_pred, slot_F1_count = 0, 0, 0, 0

    add_acc, add_num, delete_acc, delete_num = 0, 0, 0, 0

    task_acc, task_num, session_acc, session_num = 0, 0, 0, 0

    add_feedback_cost, delete_feedback_cost, all_slot, present_slot = 0, 0, 0, 0

    last_dialog_state = {}
    last_added_information = {}
    last_deleted_information = {}
    count_examples = int(max_count / 10)
    while count < max_count:
        # print(count)
        if (count % count_examples) == 0:
            print(count)
        data = data_test[count: count+num_slot]
        count += num_slot
        batch_data = {}
        for key in data[0]:
            batch_data[key] = [d[key] for d in data]
        if batch_data["turn_id"][0] == 0:
            last_dialog_state = {}
            last_added_information = {}
            last_deleted_information = {}
            last_pred_state = {}
        
        all_slot += len(ALL_SLOTS)

        feedback_active = False
        if args.feedback_timing == 'turn':
            feedback_active = True
        elif args.feedback_timing == 'task':
            if batch_data["task_final"][0] == 1:
                feedback_active = True
        elif args.feedback_timing == 'session':
            if batch_data["session_final"][0] == 1:
                feedback_active = True
        else:
            print('select feedback_timing in turn, task or session')
            exit()
        
        turn_dialog_state = batch_data["turn_dialog_state"][0]
        
        if args.feedback_acquisition_strategy == 'explicit':
            last_dialog_state_1, equal = acquire_state(batch_data, model, tokenizer, device)
            pred_state, gold_state = acquire_pred_gold(args, turn_dialog_state, last_dialog_state_1, last_dialog_state, ALL_SLOTS)
            added_information = {key:gold_state.get(key) for key in gold_state.keys() if pred_state.get(key) != gold_state.get(key) and gold_state.get(key) != 'none'}
            deleted_information = {key:pred_state.get(key) for key in gold_state.keys() if pred_state.get(key) != 'none' and gold_state.get(key) == 'none'}

            if feedback_active == True:
                present_slot += len(pred_state)
                add_feedback_cost += len(added_information)
                delete_feedback_cost += len(deleted_information)
                
            if (added_information != {} or deleted_information != {}) and feedback_active == True:
                last_dialog_state_2, equal = acquire_corrected_state(args, batch_data, added_information, deleted_information, model, tokenizer, 
                                            feedback_model, feedback_tokenizer, device)
                last_dialog_state = last_dialog_state_2
                for s in added_information:
                    add_num += 1
                    if last_dialog_state_2.get(s) == turn_dialog_state.get(s):
                        add_acc += 1
                for s in deleted_information:
                    delete_num += 1
                    if last_dialog_state_2.get(s) == turn_dialog_state.get(s):
                        delete_acc += 1
            else:
                last_dialog_state = last_dialog_state_1
        
        elif args.feedback_acquisition_strategy == 'implicit':
            if feedback_active == True:
                present_slot += len(last_pred_state)
                add_feedback_cost += len(last_added_information)
                delete_feedback_cost += len(last_deleted_information)
            if (last_added_information != {} or last_deleted_information != {}) and feedback_active == True:
                last_dialog_state_2, equal = acquire_corrected_state(args, batch_data, last_added_information, last_deleted_information, model, tokenizer, 
                                            feedback_model, feedback_tokenizer, device)
                pred_state, gold_state = acquire_pred_gold(args, turn_dialog_state, last_dialog_state_2, last_dialog_state, ALL_SLOTS)
                last_dialog_state = last_dialog_state_2
                for s in last_added_information:
                    add_num += 1
                    if last_dialog_state_2.get(s) == turn_dialog_state.get(s):
                        add_acc += 1
                for s in last_deleted_information:
                    delete_num += 1
                    if last_dialog_state_2.get(s) == turn_dialog_state.get(s):
                        delete_acc += 1
            else:
                last_dialog_state_1, equal = acquire_state(batch_data, model, tokenizer, device)
                pred_state, gold_state = acquire_pred_gold(args, turn_dialog_state, last_dialog_state_1, last_dialog_state, ALL_SLOTS)
                last_dialog_state = last_dialog_state_1
            added_information = {key:gold_state.get(key) for key in gold_state.keys() if pred_state.get(key) != gold_state.get(key) and gold_state.get(key) != 'none'}
            deleted_information = {key:pred_state.get(key) for key in gold_state.keys() if pred_state.get(key) != 'none' and gold_state.get(key) == 'none'}
            last_added_information = added_information
            last_deleted_information = deleted_information
            last_pred_state = pred_state
        else:
            print('select feedback_acquisition_strategy in explicit or implicit')
            exit()
        
        pred_state = []
        for k, v in last_dialog_state.items():
            pred_state.append('-'.join([k, v]))

        if equal:
            joint_acc += 1
        
        if batch_data["task_final"][0] == 1:
            task_num += 1
            if equal:
                task_acc += 1

        if batch_data["session_final"][0] == 1:
            session_num += 1
            if equal:
                session_acc += 1
        
        turn_state = batch_data["turn_belief"][0]
        # print('turn_state', turn_state)
        # print('pred_state', pred_state)
        
        # Compute prediction slot accuracy
        temp_acc = compute_acc(set(turn_state), set(pred_state), ALL_SLOTS)
        slot_turn_acc += temp_acc

        # Compute prediction F1 score
        temp_f1, temp_r, temp_p, count_f1 = compute_prf(turn_state, pred_state)
        slot_F1_pred += temp_f1
        slot_F1_count += count_f1

        value_batch = [last_dialog_state.get(s, 'none') for s in ALL_SLOTS]

        # print(value_batch)
        for idx, value in enumerate(value_batch):
            dial_id = batch_data["ID"][idx]
            if dial_id not in predictions:
                predictions[dial_id] = {}
                predictions[dial_id]["domain"] = batch_data["domains"][idx][0]
                predictions[dial_id]["turns"] = {}
            if batch_data["turn_id"][idx] not in predictions[dial_id]["turns"]:
                predictions[dial_id]["turns"][batch_data["turn_id"][idx]] = {"turn_belief":batch_data["turn_belief"][idx], "pred_belief":[]}

            if value!="none":
                predictions[dial_id]["turns"][batch_data["turn_id"][idx]]["pred_belief"].append(str(batch_data["slot_text"][idx])+'-'+str(value))

            # analyze slot acc:
            if str(value)==str(batch_data["value_text"][idx]):
                slot_logger[str(batch_data["slot_text"][idx])][1]+=1 # hit
            slot_logger[str(batch_data["slot_text"][idx])][0]+=1 # total
        
        # sum+=1
        # if sum > 50:
        #     exit()
    # exit()

    for slot_log in slot_logger.values():
        slot_log[2] = slot_log[1]/slot_log[0]

    with open(os.path.join(save_path, f"slot_acc.json"), 'w') as f:
        json.dump(slot_logger,f, indent=4)

    with open(os.path.join(save_path, f"prediction.json"), 'w') as f:
        json.dump(predictions,f, indent=4)

    joint_acc_score, F1_score, turn_acc_score = evaluate_metrics(predictions, ALL_SLOTS)

    evaluation_metrics = {"Joint Acc":joint_acc_score, "Turn Acc":turn_acc_score, "Joint F1":F1_score}
    print(f"result:",evaluation_metrics)

    with open(os.path.join(save_path, f"result.json"), 'w') as f:
        json.dump(evaluation_metrics,f, indent=4)
    
    len_test_data = max_count / num_slot
    joint_acc_score = joint_acc / len_test_data
    turn_acc_score = slot_turn_acc / len_test_data
    slot_F1_score = slot_F1_pred / slot_F1_count
    add_acc_score = add_acc / add_num
    delete_acc_score = delete_acc / delete_num
    task_acc_score = task_acc / task_num
    session_acc_score = session_acc / session_num
    add_feedback_cost_score = add_feedback_cost / all_slot
    delete_feedback_cost_score = delete_feedback_cost / all_slot
    add_feedback_gain_score = add_acc / all_slot
    delete_feedback_gain_score = delete_acc / all_slot
    feedback_slot_acc_score = (add_acc + delete_acc) / (add_num + delete_num)

    print("------------------------------")
    print("feedback_acquisition_strategy : ", args.feedback_acquisition_strategy)
    print("feedback_content : ", args.feedback_content)
    print("feedback_timing : ", args.feedback_timing)
    print("joint accuracy : ", joint_acc_score)
    print("slot turn accuracy : ", turn_acc_score)
    print("slot turn F1: ", slot_F1_score)
    print("add slot accuracy : ", add_acc_score)
    print("delete slot accuracy : ", delete_acc_score)
    print("Task accuracy : ", task_acc_score)
    print("Session accuracy : ", session_acc_score)
    print("add_feedback_cost_score : ", add_feedback_cost_score)
    print("delete_feedback_cost_score : ", delete_feedback_cost_score)
    print("add_feedback_gain_score : ", add_feedback_gain_score)
    print("delete_feedback_gain_score : ", delete_feedback_gain_score)
    print("feedback_slot_acc_score : ", feedback_slot_acc_score)
    print("-----------------------------\n")

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MultiWOZ_2.1", help="WOZ_2.0 or MultiWOZ_2.1")
    parser.add_argument("--model_checkpoint", type=str, default="t5-small", help="Path, url or short name of the model")
    parser.add_argument("--saving_dir", type=str, default="save", help="Path for saving")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--dev_batch_size", type=int, default=8, help="Batch size for validation")
    parser.add_argument("--test_batch_size", type=int, default=8, help="Batch size for test")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=557, help="Random seed")
    parser.add_argument("--GPU", type=int, default=1, help="number of gpu to use")
    parser.add_argument("--slot_lang", type=str, default="slottype", help="use 'none', 'human', 'naive', 'value', 'question', 'slottype' slot description")

    parser.add_argument("--feedback_acquisition_strategy", default='explicit', help='explicit or implicit', type=str)
    parser.add_argument("--feedback_content", default='belief_state', help='belief_state or turn_label', type=str)
    parser.add_argument("--feedback_timing", default='session', help='turn, task or session', type=str)
    parser.add_argument("--train_feedback", default=1.0, type=float)
    args = parser.parse_args()
    args.feedback_model_name = '../' + info.model_file[args.dataset]

    if args.dataset == 'WOZ_2.0':
        data_root = 'data/WOZ_2.0'
        config_root = 'data/dataset_config'
        feedback_data_root = 'feedback_data/WOZ_2.0'
        args.train_data_path = os.path.join(data_root, 'woz_train_en.json')
        args.dev_data_path = os.path.join(data_root, 'woz_validate_en.json')
        args.test_data_path = os.path.join(data_root, 'woz_test_en.json')
        args.config_path = os.path.join(config_root, 'woz2.json')
        args.description_path = os.path.join('utils', 'slot_description_WOZ.json')
        args.feedback_data_path = os.path.join(feedback_data_root, 'add[2]_delete[1]_seed[42].json')
    elif args.dataset == 'MultiWOZ_2.1':
        data_root = 'data/MultiWOZ_2.1'
        config_root = 'data/dataset_config'
        feedback_data_root = 'feedback_data/MultiWOZ_2.1'
        args.train_data_path = os.path.join(data_root, 'train_dials.json')
        args.dev_data_path = os.path.join(data_root, 'dev_dials.json')
        args.test_data_path = os.path.join(data_root, 'test_dials.json')
        args.config_path = os.path.join(config_root, 'multiwoz21.json')
        args.description_path = os.path.join('utils', 'slot_description_MWOZ.json')
        args.feedback_data_path = os.path.join(feedback_data_root, 'add[4]_delete[2]_seed[42].json')
    else:
        print('select dataset in WOZ_2.0 and MultiWOZ_2.1')
        exit()
    
    args.mode = 'test'
    print(args)
    test(args)
