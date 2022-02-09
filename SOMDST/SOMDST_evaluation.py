# import os
# os.environ['CUDA_VISIBLE_DEVICES']="4"
from SOMDST_utils.eval_utils import compute_prf, compute_acc
from pytorch_transformers import BertTokenizer, BertConfig

from model.SOMDST import SOMDST
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
        from SOMDST_utils.WOZ_data_utils import prepare_dataset, postprocessing, state_equal, SLOT, domain2id, OP_SET
    if args.dataset == 'MultiWOZ_2.1':
        from SOMDST_utils.MultiWOZ_data_utils import prepare_dataset, postprocessing, state_equal, domain2id, OP_SET, make_slot_meta
        SLOT = make_slot_meta(args.config_path)
    
    slot_meta = SLOT
    tokenizer = BertTokenizer(args.vocab_path, do_lower_case=True)
    data = prepare_dataset(
                    args.test_data_path,
                    tokenizer,
                    slot_meta,
                    args.n_history,
                    args.max_seq_length,
                    args.op_code)

    model_config = BertConfig.from_json_file(args.bert_config_path)
    model_config.dropout = 0.1
    op2id = OP_SET[args.op_code]
    model = SOMDST(model_config, len(op2id), len(domain2id), op2id['update'])
    ckpt = torch.load(args.model_ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt)

    model.eval()
    model.to(device)
    model_evaluation(args, postprocessing, state_equal, domain2id, OP_SET, model, data, tokenizer, slot_meta, 0, args.op_code)


def model_evaluation(args, postprocessing, state_equal, domain2id, OP_SET, model, test_data, tokenizer, slot_meta, epoch, op_code='4'):
    config_path = args.config_path
    with open(config_path, "r", encoding='utf-8') as f:
        raw_config = json.load(f)
    label_maps = raw_config['label_maps']
    
    model.eval()
    op2id = OP_SET[op_code]
    id2op = {v: k for k, v in op2id.items()}
    id2domain = {v: k for k, v in domain2id.items()}

    slot_turn_acc, joint_acc, slot_F1_pred, slot_F1_count = 0, 0, 0, 0
    op_acc, op_F1, op_F1_count = 0, {k: 0 for k in op2id}, {k: 0 for k in op2id}

    results = {}
    last_dialog_state = {}
    wall_times = []
    for di, i in enumerate(test_data):
        if i.turn_id == 0:
            last_dialog_state = {}

        i.last_dialog_state = deepcopy(last_dialog_state)
        i.make_instance(tokenizer, word_dropout=0.)

        input_ids = torch.LongTensor([i.input_id]).to(device)
        input_mask = torch.FloatTensor([i.input_mask]).to(device)
        segment_ids = torch.LongTensor([i.segment_id]).to(device)
        state_position_ids = torch.LongTensor([i.slot_position]).to(device)

        start = time.perf_counter()
        MAX_LENGTH = 9
        with torch.no_grad():
            d, s, g = model(input_ids=input_ids,
                            token_type_ids=segment_ids,
                            state_positions=state_position_ids,
                            attention_mask=input_mask,
                            max_value=MAX_LENGTH)

        _, op_ids = s.view(-1, len(op2id)).max(-1)

        if g.size(1) > 0:
            generated = g.squeeze(0).max(-1)[1].tolist()
        else:
            generated = []

        pred_ops = [id2op[a] for a in op_ids.tolist()]

        generated, last_dialog_state = postprocessing(slot_meta, pred_ops, last_dialog_state,
                                                      generated, tokenizer, op_code)
        last_dialog_state, equal = state_equal(last_dialog_state, i.turn_dialog_state, slot_meta, label_maps)
        end = time.perf_counter()
        wall_times.append(end - start)
        pred_state = []
        for k, v in last_dialog_state.items():
            pred_state.append('-'.join([k, v]))

        if equal:
            joint_acc += 1
        
        key = str(i.id) + '_' + str(i.turn_id)
        results[key] = [pred_state, i.gold_state]

        # Compute prediction slot accuracy
        temp_acc = compute_acc(set(i.gold_state), set(pred_state), slot_meta)
        slot_turn_acc += temp_acc

        # Compute prediction F1 score
        temp_f1, temp_r, temp_p, count = compute_prf(i.gold_state, pred_state)
        slot_F1_pred += temp_f1
        slot_F1_count += count

    joint_acc_score = joint_acc / len(test_data)
    turn_acc_score = slot_turn_acc / len(test_data)
    slot_F1_score = slot_F1_pred / slot_F1_count
    latency = np.mean(wall_times) * 1000

    print("------------------------------")
    print("Epoch %d joint accuracy : " % epoch, joint_acc_score)
    print("Epoch %d slot turn accuracy : " % epoch, turn_acc_score)
    print("Epoch %d slot turn F1: " % epoch, slot_F1_score)
    print("Latency Per Prediction : %f ms" % latency)
    print("-----------------------------\n")
    # json.dump(results, open('preds_%d.json' % epoch, 'w'), indent=4)
    # per_domain_join_accuracy(results, slot_meta)

    scores = {'epoch': epoch, 'joint_acc': joint_acc_score,
              'slot_acc': turn_acc_score, 'slot_f1': slot_F1_score}
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='MultiWOZ_2.1', type=str)
    parser.add_argument("--vocab_path", default='bert-base-uncased/vocab.txt', type=str)
    parser.add_argument("--bert_config_path", default='bert-base-uncased/config.json', type=str)
    parser.add_argument("--model_ckpt_path", default='outputs/model_best.bin', type=str)
    parser.add_argument("--n_history", default=1, type=int)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--op_code", default="4", type=str)

    parser.add_argument("--random_seed", default=42, type=int)

    args = parser.parse_args()
    model_name = 'model_best_seed[%s].bin'% (args.random_seed)
    if args.dataset == 'WOZ_2.0':
        data_root = 'data/WOZ_2.0'
        config_root = 'data/dataset_config'
        args.train_data_path = os.path.join(data_root, 'woz_train_en.json')
        args.dev_data_path = os.path.join(data_root, 'woz_validate_en.json')
        args.test_data_path = os.path.join(data_root, 'woz_test_en.json')
        args.config_path = os.path.join(config_root, 'woz2.json')
        args.model_ckpt_path = 'outputs/SOMDST/WOZ_outputs/' + model_name
    elif args.dataset == 'MultiWOZ_2.1':
        data_root = 'data/MultiWOZ_2.1'
        config_root = 'data/dataset_config'
        args.train_data_path = os.path.join(data_root, 'train_dials.json')
        args.dev_data_path = os.path.join(data_root, 'dev_dials.json')
        args.test_data_path = os.path.join(data_root, 'test_dials.json')
        args.config_path = os.path.join(config_root, 'multiwoz21.json')
        args.model_ckpt_path = 'outputs/SOMDST/MultiWOZ_outputs/' + model_name
    else:
        print('select dataset in WOZ_2.0 and MultiWOZ_2.1')
        exit()
    main(args)
