import numpy as np
from torch.utils.data import Dataset
import torch
from .dataset_dstc2 import create_examples

SLOT=['area', 'food', 'price range']
OP  = {'none': 0, 'dontcare': 1, 'copy_value': 2, 'unpointable': 3}

def postprocessing(slot_meta, ops, last_dialog_state, generated, input_):
    gid = 0
    for st, op in zip(slot_meta, ops):
        if op == 'dontcare' :
            last_dialog_state[st] = 'dontcare'
        elif op == 'copy_value':
            g = input_[generated[gid][0]:generated[gid][1]+1]
            gen = []
            for gg in g:
                gen.append(gg)
            gen = ' '.join(gen).replace(' ##', '')
            gen = gen.replace(' : ', ':').replace('##', '')
            last_dialog_state[st] = gen
        gid += 1
    return generated, last_dialog_state

def state_equal(pred_dialog_state, gold_dialog_state, slot_meta, value_dict):
    equal = True
    for slot in slot_meta:
        pred_value = pred_dialog_state.get(slot)
        gold_value = gold_dialog_state.get(slot)
        if pred_value != gold_value:
            equal = False
            for s in value_dict:
                if pred_value in [s]+value_dict[s]:
                    for s1 in [s]+value_dict[s]:
                        if s1 == gold_value:
                            equal = True
                            pred_dialog_state[slot] = s
                            break
    return pred_dialog_state, equal


def prepare_dataset(data_path, feedback_data_path, train_feedback, tokenizer, slot_meta, max_seq_length, data_type = '', append_history=False):
    if data_type == "train":
        examples = create_examples(data_path, feedback_data_path, train_feedback, slot_meta, data_type, tokenizer)
    else:
        examples = create_examples(data_path, feedback_data_path, train_feedback, slot_meta, data_type, tokenizer, use_asr_hyp=0, exclude_unpointable=False)
    data = []
    for example in examples:
        _, dialogue_id, turn_id = example['guid'].split('-')
        dialogue_id = int(dialogue_id)
        turn_id = int(turn_id)
        turn_num = example['turn_num']
        sys_token = example['text_a']
        usr_token = example['text_b']
        class_label = example['class_label']
        sys_label = example['text_a_label']
        usr_label = example['text_b_label']
        gold_turn_label = example['gold_turn_label']
        turn_dialog_state = example['turn_dialog_state']
        gold_state = example['gold_state']
        task_final = 0
        session_final = 0
        if (turn_id + 1) == turn_num:
            task_final = 1
            session_final = 1
        # print('turn_num',turn_num)
        # print('turn_id',turn_id)
        # print('task_final',task_final)
        instance = TrainingInstance(dialogue_id, turn_id, sys_token, usr_token, class_label, sys_label, usr_label, gold_turn_label, 
        turn_dialog_state, gold_state, task_final, session_final, max_seq_length, slot_meta)
        instance.make_instance(tokenizer)
        data.append(instance)
    return data

class TrainingInstance:
    def __init__(self, ID,
                 turn_id,
                 sys_token, 
                 usr_token, 
                 class_label, 
                 sys_label, 
                 usr_label, 
                 gold_turn_label, 
                 turn_dialog_state,
                 gold_state, 
                 task_final, 
                 session_final,
                 max_seq_length, 
                 slot_meta):
        self.id = ID
        self.turn_id = turn_id
        self.sys_token = sys_token
        self.usr_token = usr_token
        self.class_label = class_label
        self.sys_label = sys_label
        self.usr_label = usr_label
        self.gold_turn_label = gold_turn_label
        self.turn_dialog_state = turn_dialog_state
        self.gold_state = gold_state
        self.task_final = task_final
        self.session_final = session_final
        self.max_seq_length = max_seq_length
        self.slot_meta = slot_meta
        self.op2id = OP

    def make_instance(self, tokenizer, max_seq_length=None, word_dropout=0., active = True):
        if max_seq_length is None:
            max_seq_length = self.max_seq_length
        
        avail_length_1 = max_seq_length - 3

        avail_length = avail_length_1 - len(self.usr_token)

        if len(self.sys_token) > avail_length:  # truncated
            avail_length = len(self.sys_token) - avail_length
            self.sys_token = self.sys_token[avail_length:]
            if active:
                for s in self.sys_label:
                    self.sys_label[s] = self.sys_label[s][avail_length:]

        if len(self.sys_token) == 0 and len(self.usr_token) > avail_length_1:
            avail_length = len(self.usr_token) - avail_length_1
            self.usr_token = self.usr_token[avail_length:]
            if active:
                for s in self.usr_label:
                    self.usr_label[s] = self.usr_label[s][avail_length:]

        diag = ["[CLS]"] + self.sys_token + ["[SEP]"] + self.usr_token + ["[SEP]"]
        drop_mask = [0] + [1] * len(self.sys_token) + [0] + [1] * len(self.usr_token) + [0]
        segment = [0] + [0] * len(self.sys_token) + [0] + [1] * len(self.usr_token) + [1]

        # word dropout
        if word_dropout > 0.:
            drop_mask = np.array(drop_mask)
            word_drop = np.random.binomial(drop_mask.astype('int64'), word_dropout)
            diag = [w if word_drop[i] == 0 else '[UNK]' for i, w in enumerate(diag)]
        input_ = diag
        segment = segment
        self.input_ = input_

        # if len(input_)>100:
        #     print(len(input_))

        self.segment_id = segment

        input_mask = [1] * len(self.input_)
        self.input_id = tokenizer.convert_tokens_to_ids(self.input_)
        if len(input_mask) < max_seq_length:
            self.input_id = self.input_id + [0] * (max_seq_length-len(input_mask))
            self.segment_id = self.segment_id + [0] * (max_seq_length-len(input_mask))
            input_mask = input_mask + [0] * (max_seq_length-len(input_mask))

        self.input_mask = input_mask
        self.op_ids = [self.op2id[self.class_label[slot]] for slot in self.slot_meta]
        if active:
            self.span_label = []
            for s in self.slot_meta:
                slot_pos = [0] + self.sys_label[s] + [0] + self.usr_label[s] + [0]
                start_pos = 0
                end_pos = 0
                for id, label in enumerate(slot_pos):
                    if label == 1:
                        if start_pos == 0:
                            start_pos = id
                        end_pos = id
                self.span_label.append([start_pos, end_pos])

class MultiWozDataset(Dataset):
    def __init__(self, data, tokenizer, slot_meta, max_seq_length, rng, word_dropout=0.1):
        self.data = data
        self.len = len(data)
        self.tokenizer = tokenizer
        self.slot_meta = slot_meta
        self.max_seq_length = max_seq_length
        self.word_dropout = word_dropout
        self.rng = rng

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.word_dropout > 0:
            self.data[idx].make_instance(self.tokenizer, word_dropout=self.word_dropout)
        return self.data[idx]

    def collate_fn(self, batch):
        input_ids = torch.tensor([f.input_id for f in batch], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in batch], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_id for f in batch], dtype=torch.long)
        op_ids = torch.tensor([f.op_ids for f in batch], dtype=torch.long)
        span_ids = torch.tensor([f.span_label for f in batch], dtype=torch.long)
        return input_ids, input_mask, segment_ids, op_ids, span_ids
