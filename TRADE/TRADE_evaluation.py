# import os
# os.environ['CUDA_VISIBLE_DEVICES']="6"
from TRADE_utils.eval_utils import compute_prf, compute_acc, compute_goal
from model.TRADE import TRADE
import torch

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
        from TRADE_utils.WOZ_data_utils import prepare_dataset, postprocessing, state_equal, SLOT, OP
    if args.dataset == 'MultiWOZ_2.1':
        from TRADE_utils.MultiWOZ_data_utils import prepare_dataset, postprocessing, state_equal, OP, make_slot_meta
        SLOT = make_slot_meta(args.config_path)
    
    slot_meta = SLOT
    train_data_raw, dev_data_raw, test_data_raw, lang = \
        prepare_dataset(train_data_path=args.train_data_path,
                        dev_data_path=args.dev_data_path,
                        test_data_path=args.test_data_path,
                        slot_meta=slot_meta)
    print("# test examples %d" % len(test_data_raw))

    op2id = OP
    model = TRADE(lang, args.hidden_size, 0, len(op2id), slot_meta)
    ckpt = torch.load(args.model_ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt)

    model.eval()
    model.to(device)

    model_evaluation_batch(args, postprocessing, state_equal, model, test_data_raw, lang, slot_meta, OP, 0)

def model_evaluation_batch(args, postprocessing, state_equal, model, test_data, lang, slot_meta, OP, epoch):
    config_path = args.config_path
    with open(config_path, "r", encoding='utf-8') as f:
        raw_config = json.load(f)
    label_maps = raw_config['label_maps']
    
    model.eval()
    op2id = OP
    id2op = {v: k for k, v in op2id.items()}

    slot_turn_acc, joint_acc, slot_F1_pred, slot_F1_count = 0, 0, 0, 0

    results = {}
    wall_times = []

    max_count = len(test_data)
    batch_size = args.batch_size
    n = 0
    while n < max_count:
        batch = test_data[n:n+batch_size]
        n = n+batch_size

        batch.sort(key=lambda x: x.input_len, reverse=True)
        input_ids = [f.input_id for f in batch]
        input_lens = [f.input_len for f in batch]
        max_input = max(input_lens)
        for idx, v in enumerate(input_ids):
            input_ids[idx] = v + [0] * (max_input - len(v))
        input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
        input_lens = torch.tensor(input_lens, dtype=torch.long).to(device)

        start = time.time()

        max_value = 9
        with torch.no_grad():
            state_scores, gen_scores = model(input_ids=input_ids,
                                            input_lens=input_lens,
                                            max_value = max_value)
        
        for i in range(state_scores.size(0)):
            # if i > 20:
            #     exit()

            s = state_scores[i]
            g = gen_scores[i]
            _, op_ids = s.view(-1, len(op2id)).max(-1)

            if g.size(0) > 0:
                generated = g.max(-1)[1].tolist()
            else:
                generated = []
        
            pred_ops = [id2op[a] for a in op_ids.tolist()]

            last_dialog_state = postprocessing(slot_meta, pred_ops, {}, generated, lang)

            last_dialog_state, equal = state_equal(last_dialog_state, batch[i].turn_dialog_state, slot_meta, label_maps)

            pred_state = []
            for k, v in last_dialog_state.items():
                pred_state.append('-'.join([k, v]))

            if equal:
                joint_acc += 1
            # else:
            #     print('\n')
            #     print('----------------------------')
            #     print('i.turn_id',batch[i].turn_id)
            #     print('i.input_',[[i, token]for i,token in enumerate(batch[i].input_)])
            #     print('gold_op',batch[i].op_ids)
            #     print('pred_op',pred_ops)
            #     print('gold_state',batch[i].gold_state)
            #     print('pred_state',pred_state)
            #     print('window_dialog_state',batch[i].window_dialog_state)
            
            key = str(batch[i].id) + '_' + str(batch[i].turn_id)
            results[key] = [pred_ops, last_dialog_state, batch[i].op_labels, batch[i].turn_dialog_state]

            # Compute prediction slot accuracy
            temp_acc = compute_acc(set(batch[i].gold_state), set(pred_state), slot_meta)
            slot_turn_acc += temp_acc

            # Compute prediction F1 score
            temp_f1, temp_r, temp_p, count = compute_prf(batch[i].gold_state, pred_state)
            slot_F1_pred += temp_f1
            slot_F1_count += count
        
        end = time.time()
        wall_times.append(end - start)

    joint_acc_score = joint_acc / len(test_data)
    turn_acc_score = slot_turn_acc / len(test_data)
    slot_F1_score = slot_F1_pred / slot_F1_count
    latency = np.sum(wall_times) * 1000 / len(test_data)

    #compute_goal(results, slot_meta)

    print("------------------------------")
    print("Epoch %d joint accuracy : " % epoch, joint_acc_score)
    print("Epoch %d slot turn accuracy : " % epoch, turn_acc_score)
    print("Epoch %d slot turn F1: " % epoch, slot_F1_score)
    print("Latency Per Prediction : %f ms" % latency)
    print("-----------------------------\n")
    #json.dump(results, open('preds_%d.json' % epoch, 'w'))
    #per_domain_join_accuracy(results, slot_meta)

    scores = {'epoch': epoch, 'joint_acc': joint_acc_score,
              'slot_acc': turn_acc_score, 'slot_f1': slot_F1_score}
    return scores


def model_evaluation(args, postprocessing, state_equal, model, test_data, lang, slot_meta, OP, epoch):
    config_path = args.config_path
    with open(config_path, "r", encoding='utf-8') as f:
        raw_config = json.load(f)
    label_maps = raw_config['label_maps']
    
    model.eval()
    op2id = OP
    id2op = {v: k for k, v in op2id.items()}

    slot_turn_acc, joint_acc, slot_F1_pred, slot_F1_count = 0, 0, 0, 0

    results = {}
    wall_times = []
    for di, i in enumerate(test_data):
        # if di > 100:
        #     exit()

        i.make_instance(lang, word_dropout=0.)
        
        input_ids = torch.LongTensor([i.input_id]).to(device)
        
        start = time.time()
        max_value = 9
        with torch.no_grad():
            s, g = model(input_ids=input_ids, input_lens=None, max_value = max_value)

        _, op_ids = s.view(-1, len(op2id)).max(-1)

        if g.size(1) > 0:
            generated = g.squeeze(0).max(-1)[1].tolist()
        else:
            generated = []
        
        pred_ops = [id2op[a] for a in op_ids.tolist()]

        last_dialog_state = postprocessing(slot_meta, pred_ops, {}, generated, lang)

        last_dialog_state, equal = state_equal(last_dialog_state, i.turn_dialog_state, slot_meta, label_maps)
        
        end = time.time()
        wall_times.append(end - start)

        pred_state = []
        for k, v in last_dialog_state.items():
            pred_state.append('-'.join([k, v]))
        
        if equal:
            joint_acc += 1
        # else:
        #     print('--------------------------')
        #     print('utter', i.utter)
        #     print('pred_ops', pred_ops)
        #     print('pred', pred_state)
        #     print('gold', i.gold_state)

        key = str(i.id) + '_' + str(i.turn_id)
        results[key] = [pred_ops, last_dialog_state, i.op_labels, i.turn_dialog_state]

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

    #compute_goal(results, slot_meta)

    print("------------------------------")
    print("Epoch %d joint accuracy : " % epoch, joint_acc_score)
    print("Epoch %d slot turn accuracy : " % epoch, turn_acc_score)
    print("Epoch %d slot turn F1: " % epoch, slot_F1_score)
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
    parser.add_argument("--hidden_size", default=400, type=float)
    parser.add_argument("--random_seed", default=42, type=int)
    parser.add_argument("--batch_size", default=32, type=int)

    args = parser.parse_args()
    model_name = 'model_best_seed[%s].bin'% (args.random_seed)
    if args.dataset == 'WOZ_2.0':
        data_root = 'data/WOZ_2.0'
        config_root = 'data/dataset_config'
        args.train_data_path = os.path.join(data_root, 'woz_train_en.json')
        args.dev_data_path = os.path.join(data_root, 'woz_validate_en.json')
        args.test_data_path = os.path.join(data_root, 'woz_test_en.json')
        args.config_path = os.path.join(config_root, 'woz2.json')
        args.model_ckpt_path = 'outputs/TRADE/WOZ_outputs/' + model_name
    elif args.dataset == 'MultiWOZ_2.1':
        data_root = 'data/MultiWOZ_2.1'
        config_root = 'data/dataset_config'
        args.train_data_path = os.path.join(data_root, 'train_dials.json')
        args.dev_data_path = os.path.join(data_root, 'dev_dials.json')
        args.test_data_path = os.path.join(data_root, 'test_dials.json')
        args.config_path = os.path.join(config_root, 'multiwoz21.json')
        args.model_ckpt_path = 'outputs/TRADE/MultiWOZ_outputs/' + model_name
    else:
        print('select dataset in WOZ_2.0 and MultiWOZ_2.1')
        exit()
    main(args)
