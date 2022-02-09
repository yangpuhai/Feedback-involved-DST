# import os
# os.environ['CUDA_VISIBLE_DEVICES']="5"
from utils.eval_utils import compute_prf, compute_acc
from pytorch_transformers import BertTokenizer, BertConfig

from model.BERTDST import BERTDST
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import random
import numpy as np
import os
import time
import argparse
import json
from copy import deepcopy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):

    if args.dataset == 'WOZ_2.0':
        from utils.WOZ_data_utils import prepare_dataset, postprocessing, state_equal, SLOT, OP
    if args.dataset == 'MultiWOZ_2.1':
        from utils.MultiWOZ_data_utils import prepare_dataset, postprocessing, state_equal, OP, make_slot_meta
        SLOT = make_slot_meta(args.config_path)
    
    slot_meta = SLOT
    tokenizer = BertTokenizer(args.vocab_path, do_lower_case=True)
    if args.dataset == 'WOZ_2.0':
        test_data_path = args.test_data_path
    elif args.dataset == 'MultiWOZ_2.1':
        test_data_path=[args.test_data_path,args.config_path]
    data = prepare_dataset(test_data_path,
                           tokenizer,
                           slot_meta,
                           args.max_seq_length,
                           'test',
                           args.append_history)

    model_config = BertConfig.from_json_file(args.bert_config_path)
    model_config.dropout = 0.1
    op2id = OP
    model = BERTDST(model_config, len(op2id), len(slot_meta))
    ckpt = torch.load(args.model_ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt)

    model.eval()
    model.to(device)

    model_evaluation(args, postprocessing, state_equal, OP, model, data, tokenizer, slot_meta, 0)


def model_evaluation(args, postprocessing, state_equal, OP, model, test_data, tokenizer, slot_meta, epoch):
    config_path = args.config_path
    with open(config_path, "r", encoding='utf-8') as f:
        raw_config = json.load(f)
    label_maps = raw_config['label_maps']

    model.eval()
    op2id = OP
    id2op = {v: k for k, v in op2id.items()}

    slot_turn_acc, joint_acc, slot_F1_pred, slot_F1_count = 0, 0, 0, 0
    final_joint_acc, final_count, final_slot_F1_pred, final_slot_F1_count = 0, 0, 0, 0
    op_acc, op_F1, op_F1_count = 0, {k: 0 for k in op2id}, {k: 0 for k in op2id}
    all_op_F1_count = {k: 0 for k in op2id}

    tp_dic = {k: 0 for k in op2id}
    fn_dic = {k: 0 for k in op2id}
    fp_dic = {k: 0 for k in op2id}

    task_acc, task_num, session_acc, session_num = 0, 0, 0, 0

    results = {}
    last_dialog_state = {}
    wall_times = []
    for di, i in enumerate(test_data):
        # if di > 50:
        #    exit()
        if i.turn_id == 0:
            last_dialog_state = {}
        
        i.make_instance(tokenizer, word_dropout=0.)

        input_ids = torch.LongTensor([i.input_id]).to(device)
        input_mask = torch.FloatTensor([i.input_mask]).to(device)
        segment_ids = torch.LongTensor([i.segment_id]).to(device)

        start = time.perf_counter()
        with torch.no_grad():
            state, span = model(input_ids=input_ids,
                            token_type_ids=segment_ids,
                            attention_mask=input_mask)

        _, op_ids = state.view(-1, len(op2id)).max(-1)

        generated = span.squeeze(0).max(-1)[1].tolist()

        pred_ops = [id2op[a] for a in op_ids.tolist()]

        generated, last_dialog_state = postprocessing(slot_meta, pred_ops, last_dialog_state, generated, i.input_)
        
        last_dialog_state, equal = state_equal(last_dialog_state, i.turn_dialog_state, slot_meta, label_maps)
        
        end = time.perf_counter()
        wall_times.append(end - start)
        pred_state = []
        for k, v in last_dialog_state.items():
            pred_state.append('-'.join([k, v]))
        
        if equal:
            joint_acc += 1
        # else:
        #     print('\n')
        #     print('----------------------------')
        #     print('i.turn_id',i.turn_id)
        #     print('i.input_',[[i, token]for i,token in enumerate(i.input_)])
        #     print('gold_op',i.op_ids)
        #     print('pred_op',pred_ops)
        #     print('gold_span',i.span_label)
        #     print('pred_span',generated)
        #     print('gold_state',i.gold_state)
        #     print('pred_state',pred_state)
            # exit()
        
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

        # Compute prediction F1 score
        temp_f1, temp_r, temp_p, count = compute_prf(i.gold_state, pred_state)
        slot_F1_pred += temp_f1
        slot_F1_count += count

        # if i.is_last_turn:
        #     final_count += 1
        #     if set(pred_state) == set(i.gold_state):
        #         final_joint_acc += 1
        #     final_slot_F1_pred += temp_f1
        #     final_slot_F1_count += count

    joint_acc_score = joint_acc / len(test_data)
    turn_acc_score = slot_turn_acc / len(test_data)
    slot_F1_score = slot_F1_pred / slot_F1_count
    task_acc_score = task_acc / task_num
    session_acc_score = session_acc / session_num
    # final_joint_acc_score = final_joint_acc / final_count
    # final_slot_F1_score = final_slot_F1_pred / final_slot_F1_count
    latency = np.mean(wall_times) * 1000

    print("------------------------------")
    print("Epoch %d joint accuracy : " % epoch, joint_acc_score)
    print("Epoch %d slot turn accuracy : " % epoch, turn_acc_score)
    print("Epoch %d slot turn F1: " % epoch, slot_F1_score)
    print("Epoch %d op hit count : " % epoch, op_F1_count)
    print("Epoch %d op all count : " % epoch, all_op_F1_count)
    print("Task accuracy : ", task_acc_score)
    print("Session accuracy : ", session_acc_score)
    # print("Final Joint Accuracy : ", final_joint_acc_score)
    # print("Final slot turn F1 : ", final_slot_F1_score)
    print("Latency Per Prediction : %f ms" % latency)
    print("-----------------------------\n")
    #json.dump(results, open('preds_%d.json' % epoch, 'w'))
    #per_domain_join_accuracy(results, slot_meta)

    scores = {'epoch': epoch, 'joint_acc': joint_acc_score,
              'slot_acc': turn_acc_score, 'slot_f1': slot_F1_score}
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='MultiWOZ_2.1', type=str)
    parser.add_argument("--vocab_path", default='bert-base-uncased/vocab.txt', type=str)
    parser.add_argument("--bert_config_path", default='bert-base-uncased/config.json', type=str)
    parser.add_argument("--random_seed", default=42, type=int)
    parser.add_argument("--n_epochs", default=200, type=int)
    parser.add_argument("--append_history", default=True)

    parser.add_argument("--max_seq_length", default=150, type=int)

    args = parser.parse_args()
    model_name = 'model_best_epoch[%s]_maxlen[%s]_seed[%s].bin'% (str(args.n_epochs), str(args.max_seq_length), args.random_seed)
    if args.dataset == 'WOZ_2.0':
        data_root = 'data/WOZ_2.0'
        config_root = 'data/dataset_config'
        args.train_data_path = os.path.join(data_root, 'woz_train_en.json')
        args.dev_data_path = os.path.join(data_root, 'woz_validate_en.json')
        args.test_data_path = os.path.join(data_root, 'woz_test_en.json')
        args.config_path = os.path.join(config_root, 'woz2.json')
        args.model_ckpt_path = 'outputs/BERTDST/WOZ_outputs/' + model_name
    elif args.dataset == 'MultiWOZ_2.1':
        data_root = 'data/MultiWOZ_2.1'
        config_root = 'data/dataset_config'
        args.train_data_path = os.path.join(data_root, 'train_dials.json')
        args.dev_data_path = os.path.join(data_root, 'dev_dials.json')
        args.test_data_path = os.path.join(data_root, 'test_dials.json')
        args.config_path = os.path.join(config_root, 'multiwoz21.json')
        args.model_ckpt_path = 'outputs/BERTDST/MultiWOZ_outputs/' + model_name
    else:
        print('select dataset in WOZ_2.0 and MultiWOZ_2.1')
        exit()
    main(args)
