import json
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import ast
from tqdm import tqdm
import os
import random
from functools import partial
# from utils.fix_label import fix_general_label_error
from collections import OrderedDict

def fix_general_label_error(slot, value):

    GENERAL_TYPO = {
        # type
        "guesthouse":"guest house", "guesthouses":"guest house", "guest":"guest house", "mutiple sports":"multiple sports", 
        "sports":"multiple sports", "mutliple sports":"multiple sports","swimmingpool":"swimming pool", "concerthall":"concert hall", 
        "concert":"concert hall", "pool":"swimming pool", "night club":"nightclub", "mus":"museum", "ol":"architecture", 
        "colleges":"college", "coll":"college", "architectural":"architecture", "musuem":"museum", "churches":"church",
        # area
        "center":"centre", "center of town":"centre", "near city center":"centre", "in the north":"north", "cen":"centre", "east side":"east", 
        "east area":"east", "west part of town":"west", "ce":"centre",  "town center":"centre", "centre of cambridge":"centre", 
        "city center":"centre", "the south":"south", "scentre":"centre", "town centre":"centre", "in town":"centre", "north part of town":"north", 
        "centre of town":"centre", "cb30aq": "none",
        # price
        "mode":"moderate", "moderate -ly": "moderate", "mo":"moderate", 
        # day
        "next friday":"friday", "monda": "monday", 
        # parking
        "free parking":"yes",
        # internet
        "free internet":"yes",
        # star
        "4 star":"4", "4 stars":"4", "0 star rarting":"none",
        # others 
        "y":"yes", "any":"do not care", "n":"no", "does not care":"do not care", "dontcare": "do not care" , "not men":"none", "not":"none", "not mentioned":"none",
        '':"none", "not mendtioned":"none", "3 .":"3", "does not":"no", "fun":"none", "art":"none",  
        }

    if value in GENERAL_TYPO.keys():
        value = GENERAL_TYPO[value]

    # miss match slot and value 
    if  slot == "hotel-type" and value in ["nigh", "moderate -ly priced", "bed and breakfast", "centre", "venetian", "intern", "a cheap -er hotel"] or \
        slot == "hotel-internet" and value == "4" or \
        slot == "hotel-pricerange" and value == "2" or \
        slot == "attraction-type" and value in ["gastropub", "la raza", "galleria", "gallery", "science", "m"] or \
        "area" in slot and value in ["moderate"] or \
        "day" in slot and value == "t":
        value = "none"
    elif slot == "hotel-type" and value in ["hotel with free parking and free wifi", "4", "3 star hotel"]:
        value = "hotel"
    elif slot == "hotel-stars" and value == "3 star hotel":
        value = "3"
    elif "area" in slot:
        if value == "no": value = "north"
        elif value == "we": value = "west"
        elif value == "cent": value = "centre"
    elif "day" in slot:
        if value == "we": value = "wednesday"
        elif value == "no": value = "none"
    elif "price" in slot and value == "ch":
        value = "cheap"
    elif "internet" in slot and value == "free":
        value = "yes"

    # some out-of-define classification slot values
    if  slot == "restaurant-area" and value in ["stansted airport", "cambridge", "silver street"] or \
        slot == "attraction-area" and value in ["norwich", "ely", "museum", "same area as hotel"]:
        value = "none"

    return value

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]

MWOZ_SLOT_MAPS = {
    "arriveby": "arrive",
    "leaveat": "leave",
    "book day": "day",
    "book people": "people",
    "book stay": "stay",
    "book time": "time",
}

WOZ_SLOT_MAPS = {
    "price range": "pricerange"
}

ONTOLOGY = {
    "attraction":set(["area", "name", "type"]),
    "hotel": set(["area","book day","book people", "book stay", "internet", "name","parking", "price range", "stars","type"]),
    "restaurant": set(["area","book day","book people","book time","food","name","price range"]),
    "taxi": set(["arrive by","departure", "destination", "leave at"]),
    "train":set(["arrive by","book people","day","departure","destination","leave at"])
}

random.seed(42)

class DSTDataset(Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, args):
        """Reads source and target sequences from txt files."""
        self.data = data
        self.args = args

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item_info = self.data[index]
        return item_info

    def __len__(self):
        return len(self.data)

def linear_turn_label_MWOZ(turn_label, dataset, utter):

    domain_slot_value_maps = {}
    if isinstance(turn_label, dict):
        turn_label = turn_label.items()
    for (sub_domain,value) in turn_label:
        value = fix_general_label_error(sub_domain,value)
        if(value=="none"):
            continue
        cur_domain,slot_name = sub_domain.split("-")
        if dataset != 'test':
            if slot_name in ["name", "departure", "destination"] and value != 'do not care':
                if value not in utter:
                    continue
        if(cur_domain not in EXPERIMENT_DOMAINS):
            return domain_slot_value_maps

        if(slot_name in MWOZ_SLOT_MAPS):
            slot_name = MWOZ_SLOT_MAPS[slot_name]

        if(cur_domain not in domain_slot_value_maps):
            domain_slot_value_maps[cur_domain] = [[slot_name,value]]
        else:
            domain_slot_value_maps[cur_domain].append([slot_name,value])
            
    return domain_slot_value_maps

def read_data_MWOZ(args, path_name, tokenizer, dataset=None):
    print(("Reading all files from {}".format(path_name)))
    data = []
    data_bs2ut = []
    data_ut2bs = []
    # read files
    with open(path_name) as f:
        dials = json.load(f)
        for dial_dict in dials:
            # Reading data
            for ti, turn in enumerate(dial_dict["dialogue"]):
                if(turn["domain"] not in EXPERIMENT_DOMAINS):
                    continue # We skip turns that doesn't appear in EXPERIMENT_DOMAINS
                domain_slot_value_maps = linear_turn_label_MWOZ(turn["turn_label"], dataset, turn["transcript"])
                if domain_slot_value_maps == {}:
                    continue
                domain_slot_value_list = []
                for key , values in domain_slot_value_maps.items():
                    domaininfo = '[' + key + ']'
                    for name , value in values:
                        domaininfo += " "+ name + " " + value
                    domain_slot_value_list.append(domaininfo)
                domain_slot_value_str = " ".join(domain_slot_value_list)
                prefix_text_1 = 'translate belief state to dialogue: '
                prefix_text_2 = 'translate dialogue to belief state: '        
                input_text_1 = prefix_text_1 + domain_slot_value_str + f" {tokenizer.eos_token}"
                output_text_1 = turn["transcript"] + f" {tokenizer.eos_token}"
                data_detail = {
                            "dialogue_idx": dial_dict["dialogue_idx"],
                            "turn_idx": turn["turn_idx"],
                            "turn_domain": turn["domain"],
                            "system":turn["system_transcript"], 
                            "user":turn["transcript"],
                            "domain_slot_value_maps":domain_slot_value_maps,
                            "domain_slot_value_str":domain_slot_value_str,
                            "prefix_text":prefix_text_1,
                            "intput_text":input_text_1,
                            "output_text":output_text_1
                            }
                data.append(data_detail)
                if dataset == 'test':
                    data_bs2ut.append(data_detail)

                input_text_2 = prefix_text_2 + turn["transcript"] + f" {tokenizer.eos_token}"
                output_text_2 = domain_slot_value_str + f" {tokenizer.eos_token}"

                data_detail = {
                            "dialogue_idx": dial_dict["dialogue_idx"],
                            "turn_idx": turn["turn_idx"],
                            "turn_domain": turn["domain"],
                            "system":turn["system_transcript"], 
                            "user":turn["transcript"],
                            "domain_slot_value_maps":domain_slot_value_maps,
                            "domain_slot_value_str":domain_slot_value_str,
                            "prefix_text":prefix_text_2,
                            "intput_text":input_text_2,
                            "output_text":output_text_2
                            }
                data.append(data_detail)
                if dataset == 'test':
                    data_ut2bs.append(data_detail)
    # print(len(data))
    # for idx in range(10):
    #     print(data[idx])
    if dataset == 'test':
        return [data_bs2ut, data_ut2bs]
    return data

def linear_turn_label_WOZ(turn_label):

    domain_slot_value_maps = {}
    if isinstance(turn_label, dict):
        turn_label = turn_label.items()
    for (slot,value) in turn_label:
        if(value=="none"):
            continue
        if slot not in ["food", "area", "price range"]:
            continue
        if(slot in WOZ_SLOT_MAPS):
            slot = WOZ_SLOT_MAPS[slot]
        if "restaurant" not in domain_slot_value_maps:
            domain_slot_value_maps["restaurant"] = []
        domain_slot_value_maps["restaurant"].append([slot,value])
    return domain_slot_value_maps

def read_data_WOZ(args, path_name, tokenizer, dataset=None):
    print(("Reading all files from {}".format(path_name)))
    data = []
    data_bs2ut = []
    data_ut2bs = []
    domain_counter = {}
    # read files
    with open(path_name) as f:
        dials = json.load(f)
        for dial_dict in dials:
            # Reading data
            for ti, turn in enumerate(dial_dict["dialogue"]):
                domain_slot_value_maps = linear_turn_label_WOZ(turn["turn_label"])
                if domain_slot_value_maps == {}:
                    continue
                domain_slot_value_list = []
                for key , values in domain_slot_value_maps.items():
                    domaininfo = '[' + key + ']'
                    for name , value in values:
                        domaininfo += " "+ name + " " + value
                    domain_slot_value_list.append(domaininfo)
                domain_slot_value_str = " ".join(domain_slot_value_list)
                prefix_text_1 = 'translate belief state to dialogue: '
                prefix_text_2 = 'translate dialogue to belief state: '        
                input_text_1 = prefix_text_1 + domain_slot_value_str + f" {tokenizer.eos_token}"
                output_text_1 = turn["transcript"].lower() + f" {tokenizer.eos_token}"
                data_detail = {
                            "dialogue_idx": dial_dict["dialogue_idx"],
                            "turn_idx": turn["turn_idx"],
                            "system":turn["system_transcript"], 
                            "user":turn["transcript"],
                            "domain_slot_value_maps":domain_slot_value_maps,
                            "domain_slot_value_str":domain_slot_value_str,
                            "prefix_text":prefix_text_1,
                            "intput_text":input_text_1,
                            "output_text":output_text_1
                            }
                data.append(data_detail)
                if dataset == 'test':
                    data_bs2ut.append(data_detail)

                input_text_2 = prefix_text_2 + turn["transcript"].lower() + f" {tokenizer.eos_token}"
                output_text_2 = domain_slot_value_str + f" {tokenizer.eos_token}"

                data_detail = {
                            "dialogue_idx": dial_dict["dialogue_idx"],
                            "turn_idx": turn["turn_idx"],
                            "system":turn["system_transcript"], 
                            "user":turn["transcript"],
                            "domain_slot_value_maps":domain_slot_value_maps,
                            "domain_slot_value_str":domain_slot_value_str,
                            "prefix_text":prefix_text_2,
                            "intput_text":input_text_2,
                            "output_text":output_text_2
                            }
                data.append(data_detail)
                if dataset == 'test':
                    data_ut2bs.append(data_detail)
    # print(len(data))
    # for idx in range(10):
    #     print(data[idx])
    if dataset == 'test':
        return [data_bs2ut, data_ut2bs]
    return data

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
    if args['dataset'] == "MultiWOZ_2.2":
        path_train = 'data/MultiWOZ_2.2/train_dials.json'
        path_dev = 'data/MultiWOZ_2.2/dev_dials.json'
        path_test = 'data/MultiWOZ_2.2/test_dials.json'
    elif args['dataset'] == "WOZ_2.0":
        path_train = 'data/WOZ_2.0/woz_train_en.json'
        path_dev = 'data/WOZ_2.0/woz_validate_en.json'
        path_test = 'data/WOZ_2.0/woz_test_en.json'
    else:
        print('Please select dataset in MultiWOZ_2.2 and WOZ_2.0')
        exit()

    if args["mode"]=="test":
        if args['dataset'] == "MultiWOZ_2.2":
            data_test = read_data_MWOZ(args, path_test, tokenizer, "test")
        elif args['dataset'] == "WOZ_2.0":
            data_test = read_data_WOZ(args, path_test, tokenizer, "test")
        print('test_examples:', len(data_test[0]) + len(data_test[1]))
        test_dataset_bs2ut = DSTDataset(data_test[0], args)
        test_dataset_ut2bs = DSTDataset(data_test[1], args)
        test_loader_bs2ut = DataLoader(test_dataset_bs2ut, batch_size=args["test_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)
        test_loader_ut2bs = DataLoader(test_dataset_ut2bs, batch_size=args["test_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)
        return test_loader_bs2ut, test_loader_ut2bs

    if args['dataset'] == "MultiWOZ_2.2":
        data_train = read_data_MWOZ(args, path_train, tokenizer, "train")
        data_dev = read_data_MWOZ(args, path_dev, tokenizer, "dev")
        data_test = read_data_MWOZ(args, path_test, tokenizer, "test")
    elif args['dataset'] == "WOZ_2.0":
        data_train = read_data_WOZ(args, path_train, tokenizer, "train")
        data_dev = read_data_WOZ(args, path_dev, tokenizer, "dev")
        data_test = read_data_WOZ(args, path_test, tokenizer, "test")
    print('train_examples:', len(data_train))
    print('dev_examples:', len(data_dev))
    print('test_examples:', len(data_test[0]) + len(data_test[1]))

    train_dataset = DSTDataset(data_train, args)
    dev_dataset = DSTDataset(data_dev, args)
    test_dataset_bs2ut = DSTDataset(data_test[0], args)
    test_dataset_ut2bs = DSTDataset(data_test[1], args)

    train_loader = DataLoader(train_dataset, batch_size=args["train_batch_size"], shuffle=True, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)
    dev_loader = DataLoader(dev_dataset, batch_size=args["dev_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)
    test_loader_bs2ut = DataLoader(test_dataset_bs2ut, batch_size=args["test_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)
    test_loader_ut2bs = DataLoader(test_dataset_ut2bs, batch_size=args["test_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)

    return train_loader, dev_loader, [test_loader_bs2ut,test_loader_ut2bs]
