import os
os.environ['CUDA_VISIBLE_DEVICES']="1"
from TRADE_utils.eval_utils import compute_prf, compute_acc, compute_goal
from model.TRADE import TRADE
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

import random
import numpy as np
import os
import time
import argparse
import json
from copy import deepcopy

import sys
sys.path.append("..")
# print(sys.path)
import info

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):

    if args.dataset == 'WOZ_2.0':
        from TRADE_utils.WOZ_data_utils import prepare_dataset, postprocessing, state_equal, SLOT, OP
    if args.dataset == 'MultiWOZ_2.1':
        from TRADE_utils.MultiWOZ_data_utils import prepare_dataset, postprocessing, state_equal, OP, make_slot_meta
        SLOT = make_slot_meta(args.config_path)
    
    slot_meta = SLOT
    train_data_raw, dev_data_raw, test_data_raw, lang = \
        prepare_dataset(train_data_path=args.train_data_path,
                        dev_data_path=args.dev_data_path,
                        test_data_path=args.test_data_path,
                        feedback_data_path = args.feedback_data_path,
                        train_feedback = args.train_feedback,
                        slot_meta=slot_meta)
    print("# test examples %d" % len(test_data_raw))

    op2id = OP
    model = TRADE(lang, args.hidden_size, 0, len(op2id), slot_meta)
    ckpt = torch.load(args.model_ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt)

    model.eval()
    model.to(device)

    feedback_model = T5ForConditionalGeneration.from_pretrained(args.feedback_model_name)
    feedback_tokenizer = T5Tokenizer.from_pretrained(args.feedback_model_name)
    feedback_model.eval()
    feedback_model.to(device)

    model_evaluation(args, postprocessing, state_equal, model, test_data_raw, lang, slot_meta, OP, feedback_model, feedback_tokenizer)

def acquire_state(postprocessing, state_equal, i, model, lang, op2id, id2op, slot_meta, label_maps):
    i.make_instance(lang, word_dropout=0.)
    input_ids = torch.LongTensor([i.input_id]).to(device)
    max_value = 9
    with torch.no_grad():
        s, g = model(input_ids=input_ids, input_lens=None, max_value = max_value)
    _, op_ids = s.view(-1, len(op2id)).max(-1)
    if g.size(1) > 0:
        generated = g.squeeze(0).max(-1)[1].tolist()
    else:
        generated = []
    pred_ops = [id2op[a] for a in op_ids.tolist()]
    last_dialog_state_2 = postprocessing(slot_meta, pred_ops, {}, generated, lang)
    last_dialog_state_2, equal = state_equal(last_dialog_state_2, i.turn_dialog_state, slot_meta, label_maps)
    return last_dialog_state_2, equal

def acquire_corrected_state(args, postprocessing, state_equal, i, added_information, deleted_information, model, lang, 
    feedback_model, feedback_tokenizer, op2id, id2op, slot_meta, label_maps):
    feedback_added_information = deepcopy(added_information)
    feedback_deleted_information = deepcopy(deleted_information)
    
    feedback = info.simulated_negative_feedback(args.dataset, 'test', feedback_added_information, feedback_deleted_information, feedback_model, feedback_tokenizer, device)
    
    i.utter = i.utter + ' ' + feedback
    # i.utter = feedback + ' ' + i.utter
    
    last_dialog_state_2, equal = acquire_state(postprocessing, state_equal, i, model, lang, op2id, id2op, slot_meta, label_maps)
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

def model_evaluation(args, postprocessing, state_equal, model, test_data, lang, slot_meta, OP, feedback_model, feedback_tokenizer):
    config_path = args.config_path
    with open(config_path, "r", encoding='utf-8') as f:
        raw_config = json.load(f)
    label_maps = raw_config['label_maps']
    
    model.eval()
    op2id = OP
    id2op = {v: k for k, v in op2id.items()}

    slot_turn_acc, joint_acc = 0, 0

    add_acc, add_num, delete_acc, delete_num = 0, 0, 0, 0

    task_acc, task_num, session_acc, session_num = 0, 0, 0, 0

    add_feedback_cost, delete_feedback_cost, all_slot, present_slot = 0, 0, 0, 0

    results = {}
    last_dialog_state = {}
    last_added_information = {}
    last_deleted_information = {}

    results = {}
    wall_times = []
    count_examples = int(len(test_data) / 10)
    for di, i in enumerate(test_data):
        if (di % count_examples) == 0:
            print(di)
        if i.turn_id == 0:
            last_dialog_state = {}
            last_added_information = {}
            last_deleted_information = {}
            last_pred_state = {}
        
        start = time.time()
        all_slot += len(slot_meta)
        feedback_active = False
        if args.feedback_timing == 'turn':
            feedback_active = True
        elif args.feedback_timing == 'task':
            if i.task_final == 1:
                feedback_active = True
        elif args.feedback_timing == 'session':
            if i.session_final == 1:
                feedback_active = True
        else:
            print('select feedback_timing in turn, task or session')
            exit()

        if args.feedback_acquisition_strategy == 'explicit':
            last_dialog_state_1, equal = acquire_state(postprocessing, state_equal, i, model, lang, op2id, id2op, slot_meta, label_maps)
            pred_state, gold_state = acquire_pred_gold(args, i.turn_dialog_state, last_dialog_state_1, last_dialog_state, slot_meta)
            # print('last_dialog_state',last_dialog_state)
            # print('last_dialog_state_1',last_dialog_state_1)
            # print('i.turn_dialog_state',i.turn_dialog_state)
            added_information = {key:gold_state.get(key) for key in gold_state.keys() if pred_state.get(key) != gold_state.get(key) and gold_state.get(key) != 'none'}
            deleted_information = {key:pred_state.get(key) for key in gold_state.keys() if pred_state.get(key) != 'none' and gold_state.get(key) == 'none'}

            if feedback_active == True:
                present_slot += len(pred_state)
                add_feedback_cost += len(added_information)
                delete_feedback_cost += len(deleted_information)
                
            if (added_information != {} or deleted_information != {}) and feedback_active == True:
                # print('-------------------------------')
                # print('ID', i.id)
                # print('turn_id', i.turn_id)
                # print('added_information',added_information)
                # print('deleted_information',deleted_information)
                last_dialog_state_2, equal = acquire_corrected_state(args, postprocessing, state_equal, i, added_information, deleted_information, model, lang, 
                                            feedback_model, feedback_tokenizer, op2id, id2op, slot_meta, label_maps)
                last_dialog_state = last_dialog_state_2
                for s in added_information:
                    add_num += 1
                    if last_dialog_state_2.get(s) == i.turn_dialog_state.get(s):
                        add_acc += 1
                for s in deleted_information:
                    delete_num += 1
                    if last_dialog_state_2.get(s) == i.turn_dialog_state.get(s):
                        delete_acc += 1
                
            else:
                last_dialog_state = last_dialog_state_1
        
        elif args.feedback_acquisition_strategy == 'implicit':
            if feedback_active == True:
                present_slot += len(last_pred_state)
                add_feedback_cost += len(last_added_information)
                delete_feedback_cost += len(last_deleted_information)
            if (last_added_information != {} or last_deleted_information != {}) and feedback_active == True:
                last_dialog_state_2, equal = acquire_corrected_state(args, postprocessing, state_equal, i, last_added_information, last_deleted_information, model, lang, 
                                            feedback_model, feedback_tokenizer, op2id, id2op, slot_meta, label_maps)
                pred_state, gold_state = acquire_pred_gold(args, i.turn_dialog_state, last_dialog_state_2, last_dialog_state, slot_meta)
                last_dialog_state = last_dialog_state_2
                for s in last_added_information:
                    add_num += 1
                    if last_dialog_state_2.get(s) == i.turn_dialog_state.get(s):
                        add_acc += 1
                for s in last_deleted_information:
                    delete_num += 1
                    if last_dialog_state_2.get(s) == i.turn_dialog_state.get(s):
                        delete_acc += 1
            else:
                last_dialog_state_1, equal = acquire_state(postprocessing, state_equal, i, model, lang, op2id, id2op, slot_meta, label_maps)
                pred_state, gold_state = acquire_pred_gold(args, i.turn_dialog_state, last_dialog_state_1, last_dialog_state, slot_meta)
                last_dialog_state = last_dialog_state_1
            added_information = {key:gold_state.get(key) for key in gold_state.keys() if pred_state.get(key) != gold_state.get(key) and gold_state.get(key) != 'none'}
            deleted_information = {key:pred_state.get(key) for key in gold_state.keys() if pred_state.get(key) != 'none' and gold_state.get(key) == 'none'}
            last_added_information = added_information
            last_deleted_information = deleted_information
            last_pred_state = pred_state
        else:
            print('select feedback_acquisition_strategy in explicit or implicit')
            exit()
        
        end = time.time()
        wall_times.append(end - start)

        pred_state = []
        for k, v in last_dialog_state.items():
            pred_state.append('-'.join([k, v]))
        
        if equal:
            joint_acc += 1

        if i.task_final == 1:
            task_num += 1
            if equal:
                task_acc += 1

        if i.session_final == 1:
            session_num += 1
            if equal:
                session_acc += 1

        key = str(i.id) + '_' + str(i.turn_id)
        results[key] = [pred_state, i.gold_state]

        # Compute prediction slot accuracy
        temp_acc = compute_acc(set(i.gold_state), set(pred_state), slot_meta)
        slot_turn_acc += temp_acc

    joint_acc_score = joint_acc / len(test_data)
    turn_acc_score = slot_turn_acc / len(test_data)
    add_acc_score = add_acc / add_num
    delete_acc_score = delete_acc / delete_num
    task_acc_score = task_acc / task_num
    session_acc_score = session_acc / session_num
    add_feedback_cost_score = add_feedback_cost / all_slot
    delete_feedback_cost_score = delete_feedback_cost / all_slot
    add_feedback_gain_score = add_acc / all_slot
    delete_feedback_gain_score = delete_acc / all_slot
    feedback_slot_acc_score = (add_acc + delete_acc) / (add_num + delete_num)
    latency = np.mean(wall_times) * 1000

    print("------------------------------")
    print("feedback_acquisition_strategy : ", args.feedback_acquisition_strategy)
    print("feedback_content : ", args.feedback_content)
    print("feedback_timing : ", args.feedback_timing)
    print("Joint accuracy : ", joint_acc_score)
    print("slot turn accuracy : ", turn_acc_score)
    print("add slot accuracy : ", add_acc_score)
    print("delete slot accuracy : ", delete_acc_score)
    print("Task accuracy : ", task_acc_score)
    print("Session accuracy : ", session_acc_score)
    print("add_feedback_cost_score : ", add_feedback_cost_score)
    print("delete_feedback_cost_score : ", delete_feedback_cost_score)
    print("add_feedback_gain_score : ", add_feedback_gain_score)
    print("delete_feedback_gain_score : ", delete_feedback_gain_score)
    print("feedback_slot_acc_score : ", feedback_slot_acc_score)
    print("Latency Per Prediction : %f ms" % latency)
    print("-----------------------------\n")
    #per_domain_join_accuracy(results, slot_meta)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='WOZ_2.0', type=str)
    parser.add_argument("--hidden_size", default=400, type=int)
    parser.add_argument("--random_seed", default=41, type=int)
    parser.add_argument("--train_feedback", default=1.0, type=float)

    parser.add_argument("--feedback_acquisition_strategy", default='explicit', help='explicit or implicit', type=str)
    parser.add_argument("--feedback_content", default='belief_state', help='belief_state or turn_label', type=str)
    parser.add_argument("--feedback_timing", default='turn', help='turn, task or session', type=str)

    args = parser.parse_args()
    # model_name = 'model_best_seed[%s].bin'% (args.random_seed)
    model_name = 'model_best_feedback[%s]_seed[%s].bin'% (args.train_feedback, args.random_seed)
    args.feedback_model_name = '../' + info.model_file[args.dataset]
    if args.dataset == 'WOZ_2.0':
        data_root = 'data/WOZ_2.0'
        config_root = 'data/dataset_config'
        feedback_data_root = 'feedback_data/WOZ_2.0'
        args.train_data_path = os.path.join(data_root, 'woz_train_en.json')
        args.dev_data_path = os.path.join(data_root, 'woz_validate_en.json')
        args.test_data_path = os.path.join(data_root, 'woz_test_en.json')
        args.config_path = os.path.join(config_root, 'woz2.json')
        args.feedback_data_path = os.path.join(feedback_data_root, 'add[2]_delete[1]_seed[42].json')
        args.model_ckpt_path = 'outputs/TRADE/WOZ_outputs/' + model_name
    elif args.dataset == 'MultiWOZ_2.1':
        data_root = 'data/MultiWOZ_2.1'
        config_root = 'data/dataset_config'
        feedback_data_root = 'feedback_data/MultiWOZ_2.1'
        args.train_data_path = os.path.join(data_root, 'train_dials.json')
        args.dev_data_path = os.path.join(data_root, 'dev_dials.json')
        args.test_data_path = os.path.join(data_root, 'test_dials.json')
        args.config_path = os.path.join(config_root, 'multiwoz21.json')
        args.feedback_data_path = os.path.join(feedback_data_root, 'add[4]_delete[2]_seed[42].json')
        args.model_ckpt_path = 'outputs/TRADE/MultiWOZ_outputs/' + model_name
    else:
        print('select dataset in WOZ_2.0 and MultiWOZ_2.1')
        exit()
    print(args)
    main(args)
