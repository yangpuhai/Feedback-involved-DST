import json
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import ast
from tqdm import tqdm
import os
import random
from functools import partial
from utils.fix_label import fix_general_label_error
from utils.fix_label2 import fix_general_label_error2
from collections import OrderedDict
EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]

random.seed(577)

SLOT_MAPS = {
    "arriveby": "arrive",
    "leaveat": "leave",
    "book day": "day",
    "book people": "people",
    "book stay": "stay",
    "book time": "time"}

def linear_turn_label(turn_label):

    domain_slot_value_maps = {}
    for (sub_domain,value) in turn_label.items():
        value = fix_general_label_error2(sub_domain,value)
        # if(value=="none"):
        #     continue
        cur_domain,slot_name = sub_domain.split("-")
        if(cur_domain not in EXPERIMENT_DOMAINS):
            return domain_slot_value_maps

        if(slot_name in SLOT_MAPS):
            slot_name = SLOT_MAPS[slot_name]

        if(cur_domain not in domain_slot_value_maps):
            domain_slot_value_maps[cur_domain] = [[slot_name,value]]
        else:
            domain_slot_value_maps[cur_domain].append([slot_name,value])
            
    return domain_slot_value_maps

class DSTDataset(Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, args):
        """Reads source and target sequences from txt files."""
        self.data = data
        self.args = args

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item_info = self.data[index]
        if self.args["slot_lang"] == "value":
            random.shuffle(item_info["value_list"])
            item_info["intput_text"] += " is " + " or ".join(item_info["value_list"]) + " or none?"
        return item_info

    def __len__(self):
        return len(self.data)

def process_state(state, slot_meta):
    result = {}
    for s in state:
        if s['act'] == 'inform':
            slot = s['slots'][0][0]
            if slot not in slot_meta:
                continue
            value = s['slots'][0][1]
            result[slot] = value
    return result

def read_data_WOZ(args, path_name, path_feedback, SLOTS, tokenizer, description, dataset=None):
    print(("Reading all files from {}".format(path_name)))
    data = []

    if dataset == 'train':
        with open(path_feedback) as f2:
            feedback_set = json.load(f2)
        feedback_keys = [k for k,v in feedback_set.items() if 'rate' not in k]
        random.seed(42)
        feedback_keys = random.sample(feedback_keys, int(args["train_feedback"]*len(feedback_keys)))
        random.seed()
        # print(len(feedback_keys))
        feedback_set = {k:v for k,v in feedback_set.items() if k in feedback_keys}
        # print(len(feedback_set))

    # read files
    with open(path_name) as f:
        dials = json.load(f)

        for dial_dict in dials:
            dialog_history = ""

            # Reading data
            for ti, turn in enumerate(dial_dict["dialogue"]):
                turn_id = turn["turn_idx"]

                # accumulate dialogue utterances
                dialog_history +=  (" System: " + turn["system_transcript"].lower() + " User: " + turn["transcript"].lower())
                slot_values = process_state(turn["belief_state"], SLOTS)

                feedback = False
                if dataset == 'train':
                    fid = str(dial_dict["dialogue_idx"]) + '-' + str(turn['turn_idx'])
                    if fid in feedback_set:
                        feedback = True
            
                # input: dialogue history + slot
                # output: value

                # Generate domain-dependent slot list
                slot_temp = SLOTS

                len_turns = len(dial_dict['dialogue'])
                task_final = 0
                session_final = 0
                if (turn_id + 1) == len_turns:
                    task_final = 1
                    session_final = 1

                turn_belief_list = [str(k)+'-'+str(v) for k,v in slot_values.items()]

                if feedback:
                    new_slot_values = process_state(feedback_set[fid]["belief_state"], SLOTS)
                    new_turn_belief_list = [str(k)+'-'+str(v) for k,v in new_slot_values.items()]

                for slot in slot_temp:

                    output_text = slot_values.get(slot, 'none').strip() + f" {tokenizer.eos_token}"
                    slot_text = slot
                    value_text = slot_values.get(slot, 'none').strip()

                    if args["slot_lang"]=="human":
                        slot_lang = description[slot]["description_human"]
                        input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}?"
                    elif args["slot_lang"]=="naive":
                        slot_lang = description[slot]["naive"]
                        input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}?"
                    elif args["slot_lang"]=="value":
                        slot_lang = description[slot]["naive"]
                        input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}"
                    elif args["slot_lang"]=="question":
                        slot_lang = description[slot]["question"]
                        input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}"
                    elif args["slot_lang"]=="slottype":
                        slot_lang = description[slot]["slottype"]
                        input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}?"
                    else:
                        input_text = dialog_history + f" {tokenizer.sep_token} {slot}"

                    data_detail = {
                        "ID":dial_dict["dialogue_idx"],
                        "domains":["restaurant"],
                        "turn_id":turn_id,
                        "task_final":task_final,
                        "session_final":session_final,
                        "dialog_history":dialog_history,
                        "turn_dialog_state":slot_values,
                        "turn_belief":turn_belief_list,
                        "intput_text":input_text,
                        "output_text":output_text,
                        "slot_text":slot_text,
                        "value_text":value_text,
                        "value_list":description[slot]["values"]
                        }
                    data.append(data_detail)

                    if feedback:
                        output_text = new_slot_values.get(slot, 'none').strip() + f" {tokenizer.eos_token}"
                        slot_text = slot
                        value_text = new_slot_values.get(slot, 'none').strip()

                        if args["slot_lang"]=="human":
                            slot_lang = description[slot]["description_human"]
                            input_text = dialog_history + ' ' + feedback_set[fid]["feedback"] + f" {tokenizer.sep_token} {slot_lang}?"
                        elif args["slot_lang"]=="naive":
                            slot_lang = description[slot]["naive"]
                            input_text = dialog_history + ' ' + feedback_set[fid]["feedback"] + f" {tokenizer.sep_token} {slot_lang}?"
                        elif args["slot_lang"]=="value":
                            slot_lang = description[slot]["naive"]
                            input_text = dialog_history + ' ' + feedback_set[fid]["feedback"] + f" {tokenizer.sep_token} {slot_lang}"
                        elif args["slot_lang"]=="question":
                            slot_lang = description[slot]["question"]
                            input_text = dialog_history + ' ' + feedback_set[fid]["feedback"] + f" {tokenizer.sep_token} {slot_lang}"
                        elif args["slot_lang"]=="slottype":
                            slot_lang = description[slot]["slottype"]
                            input_text = dialog_history + ' ' + feedback_set[fid]["feedback"] + f" {tokenizer.sep_token} {slot_lang}?"
                        else:
                            input_text = dialog_history + ' ' + feedback_set[fid]["feedback"] + f" {tokenizer.sep_token} {slot}"

                        data_detail = {
                            "ID":dial_dict["dialogue_idx"],
                            "domains":["restaurant"],
                            "turn_id":turn_id,
                            "task_final":task_final,
                            "session_final":session_final,
                            "dialog_history":dialog_history + ' ' + feedback_set[fid]["feedback"],
                            "turn_dialog_state":new_slot_values,
                            "turn_belief":new_turn_belief_list,
                            "intput_text":input_text,
                            "output_text":output_text,
                            "slot_text":slot_text,
                            "value_text":value_text,
                            "value_list":description[slot]["values"]
                            }
                        data.append(data_detail)

    # for idx in range(10):
    #     print(data[idx])
    return data, slot_temp


def read_data_MWOZ(args, path_name, path_feedback, SLOTS, tokenizer, description, dataset=None):
    print(("Reading all files from {}".format(path_name)))
    data = []

    if dataset == 'train':
        with open(path_feedback) as f2:
            feedback_set = json.load(f2)
        feedback_keys = [k for k,v in feedback_set.items() if 'rate' not in k]
        random.seed(42)
        feedback_keys = random.sample(feedback_keys, int(args["train_feedback"]*len(feedback_keys)))
        random.seed()
        # print(len(feedback_keys))
        feedback_set = {k:v for k,v in feedback_set.items() if k in feedback_keys}
        # print(len(feedback_set))
    
    domain_counter = {}
    # read files
    with open(path_name) as f:
        dials = json.load(f)

        for dial_dict in dials:
            dialog_history = ""

            # Counting domains
            for domain in dial_dict["domains"]:
                if domain not in EXPERIMENT_DOMAINS:
                    continue
                if domain not in domain_counter.keys():
                    domain_counter[domain] = 0
                domain_counter[domain] += 1

            # Reading data
            for ti, turn in enumerate(dial_dict["dialogue"]):
                turn_domain = turn["domain"]
                turn_id = turn["turn_idx"]
                if turn_domain not in EXPERIMENT_DOMAINS:
                    continue

                # accumulate dialogue utterances
                dialog_history +=  (" System: " + turn["system_transcript"] + " User: " + turn["transcript"])
                slot_values = process_state(turn["belief_state"], SLOTS)
                slot_values = fix_general_label_error(slot_values, SLOTS)

                feedback = False
                if dataset == 'train':
                    fid = str(dial_dict["dialogue_idx"]) + '-' + str(turn['turn_idx'])
                    if fid in feedback_set:
                        feedback = True
                # input: dialogue history + slot
                # output: value

                # Generate domain-dependent slot list
                slot_temp = SLOTS

                len_turns = len(dial_dict['dialogue'])
                task_final = 0
                session_final = 0
                if turn_id + 1 < len_turns:
                    if dial_dict["dialogue"][turn_id + 1]['domain'] != turn_domain:
                        task_final = 1
                if (turn_id + 1) == len_turns:
                    task_final = 1
                    session_final = 1

                turn_belief_list = [str(k)+'-'+str(v) for k,v in slot_values.items()]

                if feedback:
                    new_slot_values = process_state(feedback_set[fid]["belief_state"], SLOTS)
                    new_slot_values = fix_general_label_error(new_slot_values, SLOTS)
                    new_turn_belief_list = [str(k)+'-'+str(v) for k,v in new_slot_values.items()]

                for slot in slot_temp:

                    output_text = slot_values.get(slot, 'none').strip() + f" {tokenizer.eos_token}"
                    slot_text = slot
                    value_text = slot_values.get(slot, 'none').strip()

                    if args["slot_lang"]=="human":
                        slot_lang = description[slot]["description_human"]
                        input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}?"
                    elif args["slot_lang"]=="naive":
                        slot_lang = description[slot]["naive"]
                        input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}?"
                    elif args["slot_lang"]=="value":
                        slot_lang = description[slot]["naive"]
                        input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}"
                    elif args["slot_lang"]=="question":
                        slot_lang = description[slot]["question"]
                        input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}"
                    elif args["slot_lang"]=="slottype":
                        slot_lang = description[slot]["slottype"]
                        input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}?"
                    else:
                        input_text = dialog_history + f" {tokenizer.sep_token} {slot}"

                    data_detail = {
                        "ID":dial_dict["dialogue_idx"],
                        "domains":dial_dict["domains"],
                        "turn_id":turn_id,
                        "task_final":task_final,
                        "session_final":session_final,
                        "dialog_history":dialog_history,
                        "turn_dialog_state":slot_values,
                        "turn_belief":turn_belief_list,
                        "intput_text":input_text,
                        "output_text":output_text,
                        "slot_text":slot_text,
                        "value_text":value_text,
                        "value_list":description[slot]["values"]
                        }
                    data.append(data_detail)

                    if feedback:
                        output_text = new_slot_values.get(slot, 'none').strip() + f" {tokenizer.eos_token}"
                        slot_text = slot
                        value_text = new_slot_values.get(slot, 'none').strip()

                        if args["slot_lang"]=="human":
                            slot_lang = description[slot]["description_human"]
                            input_text = dialog_history + ' ' + feedback_set[fid]["feedback"] + f" {tokenizer.sep_token} {slot_lang}?"
                        elif args["slot_lang"]=="naive":
                            slot_lang = description[slot]["naive"]
                            input_text = dialog_history + ' ' + feedback_set[fid]["feedback"] + f" {tokenizer.sep_token} {slot_lang}?"
                        elif args["slot_lang"]=="value":
                            slot_lang = description[slot]["naive"]
                            input_text = dialog_history + ' ' + feedback_set[fid]["feedback"] + f" {tokenizer.sep_token} {slot_lang}"
                        elif args["slot_lang"]=="question":
                            slot_lang = description[slot]["question"]
                            input_text = dialog_history + ' ' + feedback_set[fid]["feedback"] + f" {tokenizer.sep_token} {slot_lang}"
                        elif args["slot_lang"]=="slottype":
                            slot_lang = description[slot]["slottype"]
                            input_text = dialog_history + ' ' + feedback_set[fid]["feedback"] + f" {tokenizer.sep_token} {slot_lang}?"
                        else:
                            input_text = dialog_history + ' ' + feedback_set[fid]["feedback"] + f" {tokenizer.sep_token} {slot}"

                        data_detail = {
                            "ID":dial_dict["dialogue_idx"],
                            "domains":["restaurant"],
                            "turn_id":turn_id,
                            "task_final":task_final,
                            "session_final":session_final,
                            "dialog_history":dialog_history + ' ' + feedback_set[fid]["feedback"],
                            "turn_dialog_state":new_slot_values,
                            "turn_belief":new_turn_belief_list,
                            "intput_text":input_text,
                            "output_text":output_text,
                            "slot_text":slot_text,
                            "value_text":value_text,
                            "value_list":description[slot]["values"]
                            }
                        data.append(data_detail)
    # for idx in range(10):
    #     print(data[idx])
    print("domain_counter", domain_counter)
    return data, slot_temp

def get_slot_information(config_path):
    with open(config_path, "r", encoding='utf-8') as f:
        raw_config = json.load(f)
    slot_list = raw_config['slots']
    return slot_list

def collate_fn(data, tokenizer):
    batch_data = {}
    for key in data[0]:
        batch_data[key] = [d[key] for d in data]

    input_batch = tokenizer(batch_data["intput_text"], padding=True, return_tensors="pt", add_special_tokens=False, verbose=False)
    batch_data["encoder_input"] = input_batch["input_ids"]
    batch_data["attention_mask"] = input_batch["attention_mask"]
    output_batch = tokenizer(batch_data["output_text"], padding=True, return_tensors="pt", add_special_tokens=False, return_attention_mask=False)
    # replace the padding id to -100 for cross-entropy
    output_batch['input_ids'].masked_fill_(output_batch['input_ids']==tokenizer.pad_token_id, -100)
    batch_data["decoder_output"] = output_batch['input_ids']

    return batch_data


def prepare_data(args, tokenizer):
    path_train = args["train_data_path"]
    path_dev = args["dev_data_path"]
    path_test = args["test_data_path"]
    path_feedback = args["feedback_data_path"]

    ALL_SLOTS = get_slot_information(args["config_path"])
    description = json.load(open(args["description_path"], 'r'))
    print(ALL_SLOTS)

    if args["mode"] == 'test':
        if args["dataset"] == 'WOZ_2.0':
            data_test, ALL_SLOTS = read_data_WOZ(args, path_test, path_feedback, ALL_SLOTS, tokenizer, description, "test")
        elif args["dataset"] == 'MultiWOZ_2.1':
            data_test, ALL_SLOTS = read_data_MWOZ(args, path_test, path_feedback, ALL_SLOTS, tokenizer, description, "test")
        print('test_examples:', len(data_test))
        test_dataset = DSTDataset(data_test, args)
        test_loader = DataLoader(test_dataset, batch_size=args["test_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)
        return data_test, test_loader, ALL_SLOTS

    if args["dataset"] == 'WOZ_2.0':
        data_train, _ = read_data_WOZ(args, path_train, path_feedback, ALL_SLOTS, tokenizer, description, "train")
        data_dev, _ = read_data_WOZ(args, path_dev, path_feedback, ALL_SLOTS, tokenizer, description, "dev")
        data_test, ALL_SLOTS = read_data_WOZ(args, path_test, path_feedback, ALL_SLOTS, tokenizer, description, "test")
    elif args["dataset"] == 'MultiWOZ_2.1':
        data_train, _ = read_data_MWOZ(args, path_train, path_feedback, ALL_SLOTS, tokenizer, description, "train")
        data_dev, _ = read_data_MWOZ(args, path_dev, path_feedback, ALL_SLOTS, tokenizer, description, "dev")
        data_test, ALL_SLOTS = read_data_MWOZ(args, path_test, path_feedback, ALL_SLOTS, tokenizer, description, "test")
    print('train_examples:', len(data_train))
    print('dev_examples:', len(data_dev))
    print('test_examples:', len(data_test))
    # exit()

    train_dataset = DSTDataset(data_train, args)
    dev_dataset = DSTDataset(data_dev, args)
    test_dataset = DSTDataset(data_test, args)

    train_loader = DataLoader(train_dataset, batch_size=args["train_batch_size"], shuffle=True, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)
    test_loader = DataLoader(test_dataset, batch_size=args["test_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)
    dev_loader = DataLoader(dev_dataset, batch_size=args["dev_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)

    return train_loader, dev_loader, test_loader, ALL_SLOTS
