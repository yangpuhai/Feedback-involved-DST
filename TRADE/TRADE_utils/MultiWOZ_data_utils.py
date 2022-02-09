import numpy as np
import json
from torch.utils.data import Dataset
import torch
from copy import deepcopy
from .fix_label import fix_general_label_error
from .create_data import normalize
import random

flatten = lambda x: [i for s in x for i in s]
EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]
OP = {'ptr': 0, 'none': 1, 'dontcare': 2}

PAD_token = 0
UNK_token = 1
SOS_token = 2
EOS_token = 3

class Lang:
    def __init__(self):
        self.word2index = {}
        self.index2word = {PAD_token: "[PAD]", UNK_token: '[UNK]', SOS_token: "[SOS]", EOS_token: "[EOS]"}
        self.n_words = len(self.index2word)  # Count default tokens
        self.word2index = dict([(v, k) for k, v in self.index2word.items()])

    def index_words(self, sent, stype):
        if stype == 'list':
            for word in sent:
                self.index_word(word)
        if stype == 'str':
            for word in sent.split(" "):
                self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def convert_tokens_to_ids(self, sent):
        result=[]
        for t in sent:
            if t in self.word2index:
                result.append(self.word2index[t])
            else:
                result.append(UNK_token)
        return result

    def convert_ids_to_tokens(self, sent):
        result = []
        for t in sent:
            result.append(self.index2word[t])
        return result
    
    def tokenize(self, str1):
        return str1.split()

def make_turn_label(turn_dialog_state, slot_meta):
    op_labels = []
    generate_y = []
    for slot in slot_meta:
        turn_value = turn_dialog_state.get(slot)
        if turn_value == 'dontcare':
            op_labels.append('dontcare')
            generate_y.append(['[PAD]'])
        elif turn_value is None:
            op_labels.append('none')
            generate_y.append(['[PAD]'])
        else:
            op_labels.append('ptr')
            generate_y.append([v for v in turn_value.split()] + ['[EOS]'])

    gold_state = [str(k) + '-' + str(v) for k, v in turn_dialog_state.items()]

    return op_labels, generate_y, gold_state


def postprocessing(slot_meta, ops, last_dialog_state, generated, lang):
    gid = 0
    for st, op in zip(slot_meta, ops):
        if op == 'dontcare':
            last_dialog_state[st] = 'dontcare'
        elif op == 'ptr':
            g = lang.convert_ids_to_tokens(generated[gid])
            gen = []
            for gg in g:
                if gg == '[EOS]':
                    break
                gen.append(gg)
            gen = ' '.join(gen)
            last_dialog_state[st] = gen
        gid += 1
    return last_dialog_state

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

def make_slot_meta(config_path):
    with open(config_path, "r", encoding='utf-8') as f:
        raw_config = json.load(f)
    slot_list = raw_config['slots']
    return slot_list

def load_data(train_dials, feedback_set, lang, slot_meta, data_type = ''):
    # load training data
    data = []
    for dial_dict in train_dials:
        dialog_history = []
        for ti, turn in enumerate(dial_dict["dialogue"]):
            turn_domain = turn["domain"]
            turn_id = turn['turn_idx']
            if turn_domain not in EXPERIMENT_DOMAINS:
                continue

            system_uttr = normalize(turn['system_transcript'].strip(), False)
            user_uttr = normalize(turn['transcript'].strip(), False)
            if system_uttr == '':
                turn_uttr = '; ' + user_uttr
            else:
                turn_uttr = system_uttr + ' ; ' + user_uttr
            dialog_history.append(turn_uttr)
            turn_dialog_state = fix_general_label_error(turn["belief_state"], False, slot_meta)

            turn_dialog_state = {k: v for k, v in turn_dialog_state.items() if k in slot_meta}

            keys = list(turn_dialog_state.keys())
            for k in keys:
                if turn_dialog_state.get(k) == 'none':
                    turn_dialog_state.pop(k)

            len_turns = len(dial_dict['dialogue'])
            dialogue_id = dial_dict["dialogue_idx"]

            task_final = 0
            session_final = 0
            if turn_id + 1 < len_turns:
                if dial_dict["dialogue"][turn_id + 1]['domain'] != turn_domain:
                    task_final = 1
            if (turn_id + 1) == len_turns:
                task_final = 1
                session_final = 1
            
            op_labels, generate_y, gold_state = make_turn_label(turn_dialog_state, slot_meta)
            instance = TrainingInstance(dialogue_id, turn_id, " ; ".join(dialog_history), 
            deepcopy(turn_dialog_state), op_labels, generate_y, gold_state, task_final, session_final, slot_meta)
            instance.make_instance(lang)
            data.append(instance)

            if data_type == 'train':
                fid = str(dial_dict["dialogue_idx"]) + '-' + str(turn['turn_idx'])
                if fid in feedback_set:
                    feedback = normalize(feedback_set[fid]["feedback"], False)
                    new_turn_dialog_state = fix_general_label_error(feedback_set[fid]["belief_state"], False, slot_meta)
                    new_turn_dialog_state = {k: v for k, v in new_turn_dialog_state.items() if k in slot_meta}
                    keys = list(new_turn_dialog_state.keys())
                    for k in keys:
                        if new_turn_dialog_state.get(k) == 'none':
                            new_turn_dialog_state.pop(k)
                    op_labels, generate_y, gold_state = make_turn_label(new_turn_dialog_state, slot_meta)
                    instance = TrainingInstance(dialogue_id, turn_id, " ; ".join(dialog_history) + ' ' + feedback, 
                    deepcopy(new_turn_dialog_state), op_labels, generate_y, gold_state, task_final, session_final, slot_meta)
                    instance.make_instance(lang)
                    data.append(instance)

    return data

def prepare_dataset(train_data_path, dev_data_path, test_data_path, feedback_data_path, train_feedback, slot_meta):
    train_dials = json.load(open(train_data_path))
    dev_dials = json.load(open(dev_data_path))
    test_dials = json.load(open(test_data_path))
    feedback_set = json.load(open(feedback_data_path))

    lang = Lang()
    for dial_dict in train_dials:
        for turn in dial_dict["dialogue"]:
            turn_domain = turn["domain"]
            if turn_domain not in EXPERIMENT_DOMAINS:
                continue
            system_tokens = turn['system_transcript'].strip().split(' ')
            lang.index_words(system_tokens,'list')
            user_tokens = turn['transcript'].strip().split(' ')
            lang.index_words(user_tokens,'list')
            for s in turn['belief_state']:
                lang.index_words(s["slots"][0][0].split('-')[0].split(), 'list')
                lang.index_words(s["slots"][0][0].split('-')[1].split(), 'list')
                lang.index_words(s["slots"][0][1].split(' '), 'list')

    for dial in feedback_set:
        dial_data = feedback_set[dial]
        if 'rate' not in dial:
            feedback_tokens = dial_data['feedback'].strip().split(' ')
            lang.index_words(feedback_tokens,'list')
            for s in dial_data['belief_state']:
                if s['act'] == 'inform':
                    lang.index_words(s['slots'][0][0].split(" "), 'list')
                    lang.index_words(s['slots'][0][1].split(" "), 'list')

    feedback_keys = [k for k,v in feedback_set.items() if 'rate' not in k]
    random.seed(42)
    feedback_keys = random.sample(feedback_keys, int(train_feedback*len(feedback_keys)))
    random.seed()
    # print(len(feedback_keys))
    feedback_set = {k:v for k,v in feedback_set.items() if k in feedback_keys}

    #load training data
    train_data = load_data(train_dials, feedback_set, lang, slot_meta, 'train')
    dev_data = load_data(dev_dials, feedback_set, lang, slot_meta, 'dev')
    test_data = load_data(test_dials, feedback_set, lang, slot_meta, 'test')

    return train_data, dev_data, test_data, lang


class TrainingInstance:
    def __init__(self, ID,
                 turn_id,
                 utter,
                 turn_dialog_state,
                 op_labels, 
                 generate_y, 
                 gold_state,
                 task_final, 
                 session_final,
                 slot_meta):
        self.id = ID
        self.turn_id = turn_id
        self.utter = utter
        self.turn_dialog_state = turn_dialog_state
        self.gold_state = gold_state
        self.op_labels = op_labels
        self.generate_y = generate_y
        self.task_final = task_final
        self.session_final = session_final
        self.slot_meta = slot_meta
        self.op2id = OP

    def make_instance(self, lang, word_dropout=0.):

        #process text
        diag = self.utter
        diag= diag.strip().split(" ")
        drop_mask = [1] * len(diag)
        # word dropout
        if word_dropout > 0.:
            drop_mask = np.array(drop_mask)
            word_drop = np.random.binomial(drop_mask.astype('int64'), word_dropout)
            diag = [w if word_drop[i] == 0 else '[UNK]' for i, w in enumerate(diag)]
        input_ = diag
        self.input_ = input_
        self.input_id = lang.convert_tokens_to_ids(self.input_)
        self.input_len = len(self.input_id)
        self.op_ids = [self.op2id[a] for a in self.op_labels]
        self.generate_ids = [lang.convert_tokens_to_ids(y) for y in self.generate_y]

class MultiWozDataset(Dataset):
    def __init__(self, data, lang, word_dropout=0.1):
        self.data = data
        self.len = len(data)
        self.lang = lang
        self.word_dropout = word_dropout

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.word_dropout > 0:
            self.data[idx].make_instance(self.lang, word_dropout=self.word_dropout)
        return self.data[idx]

    def collate_fn(self, batch):
        batch.sort(key=lambda x: x.input_len, reverse=True)
        input_ids = [f.input_id for f in batch]
        input_lens = [f.input_len for f in batch]
        max_input = max(input_lens)
        for idx, v in enumerate(input_ids):
            input_ids[idx] = v + [0] * (max_input - len(v))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_lens = torch.tensor(input_lens, dtype=torch.long)

        op_ids = torch.tensor([f.op_ids for f in batch], dtype=torch.long)
        gen_ids = [b.generate_ids for b in batch]
        max_update = max([len(b) for b in gen_ids])
        max_value_list=[len(b) for b in flatten(gen_ids)]
        max_value = max(max_value_list)

        for bid, b in enumerate(gen_ids):
            n_update = len(b)
            for idx, v in enumerate(b):
                b[idx] = v + [0] * (max_value - len(v))
            gen_ids[bid] = b + [[0] * max_value] * (max_update - n_update)

        gen_ids = torch.tensor(gen_ids, dtype=torch.long)

        return input_ids, op_ids, gen_ids, input_lens, max_input, max_value
