import os
import numpy as np
import torch
from tqdm import tqdm
import random
import copy
from transformers import T5ForConditionalGeneration, T5Tokenizer
import json

from info import model_file, introducer_list
from Feedback_Simulation.data_loader import linear_turn_label_MWOZ, linear_turn_label_WOZ, EXPERIMENT_DOMAINS

os.environ['CUDA_VISIBLE_DEVICES']="1"

def set_seed(args_seed, args_n_gpu):
    np.random.seed(args_seed)
    torch.manual_seed(args_seed)
    random.seed(args_seed)
    if args_n_gpu > 0:
        torch.cuda.manual_seed_all(args_seed)

def update_turn(turn, add_slot_dict, delete_slot_dict, add_success, delete_success):
    new_turn_label = []
    new_belief = []

    for (slot,value) in turn["turn_label"]:
        if slot in add_slot_dict and add_success:
            value = add_slot_dict[slot]
        if slot in delete_slot_dict and delete_success:
            continue
        new_turn_label.append([slot,value])
    
    turn_slot = [sv[0] for sv in new_turn_label]
    
    for slot,value in add_slot_dict.items():
        if slot not in turn_slot:
            new_turn_label.append([slot,value])
    
    for bs in turn["belief_state"]:
        copy_bs = copy.deepcopy(bs)
        if bs['act'] == 'inform':
            slot, value = bs["slots"][0]
            if slot in add_slot_dict and add_success:
                value = add_slot_dict[slot]
            if slot in delete_slot_dict and delete_success:
                continue
            copy_bs["slots"][0][1] = value
        new_belief.append(copy_bs)

    turn["belief_state"] = new_belief
    turn["turn_label"] = new_turn_label
    return turn

def create_and_filtering_feedback(dataset, information, SFB_model, SFB_tokenizer, device):
    feedback = []
    if dataset == 'MultiWOZ_2.1':
        domain_slot_value_maps = linear_turn_label_MWOZ(information, 'test', '')
    elif dataset == 'WOZ_2.0':
        domain_slot_value_maps = linear_turn_label_WOZ(information)
    
    for key , values in domain_slot_value_maps.items():
        domaininfo = '[' + key + ']'
        for name , value in values:
            domaininfo += " "+ name + " " + value
        domain_slot_value_str = domaininfo
        prefix_text_1 = 'translate belief state to dialogue: '    
        input_text_1 = prefix_text_1 + domain_slot_value_str + f" {SFB_tokenizer.eos_token}"
        input_batch = SFB_tokenizer([input_text_1], padding=True, return_tensors="pt", add_special_tokens=False, verbose=False)
        encoder_input = input_batch["input_ids"]
        attention_mask = input_batch["attention_mask"]
        dst_outputs = SFB_model.generate(input_ids=encoder_input.to(device),
                        attention_mask=attention_mask.to(device),
                        eos_token_id=SFB_tokenizer.eos_token_id,
                        max_length=50,
                        num_beams=5,
                        num_return_sequences=5
                        )
        value_batch = SFB_tokenizer.batch_decode(dst_outputs, skip_special_tokens=True)
        # print('value_batch', value_batch)
        prefix_text_2 = 'translate dialogue to belief state: '
        input_text_2 = [prefix_text_2 + t + f" {SFB_tokenizer.eos_token}" for t in value_batch]
        input_batch2 = SFB_tokenizer(input_text_2, padding=True, return_tensors="pt", add_special_tokens=False, verbose=False)
        encoder_input2 = input_batch2["input_ids"]
        attention_mask2 = input_batch2["attention_mask"]
        dst_outputs2 = SFB_model.generate(input_ids=encoder_input2.to(device),
                        attention_mask=attention_mask2.to(device),
                        eos_token_id=SFB_tokenizer.eos_token_id,
                        max_length=50
                        )
        value_batch2 = SFB_tokenizer.batch_decode(dst_outputs2, skip_special_tokens=True)

        str1 = ''
        for s1, s2 in zip(value_batch, value_batch2):
            if s2 == domain_slot_value_str and '?' not in s1:
                str1 = s1
                break
        # print(str1)
        feedback.append(str1)
    return feedback

def feedback_utter_generator(dataset, data_type, added_information, deleted_information, SFB_model, SFB_tokenizer, device):
    feedback1 = []
    add_success = False
    add_feedback = ''
    if added_information != {}:
        feedback1 = create_and_filtering_feedback(dataset, added_information, SFB_model, SFB_tokenizer, device)
        if '' not in feedback1:
            add_success = True
        add_feedback = ' and '.join(feedback1)

    feedback2 = []
    delete_success = False
    delete_feedback = ''
    if deleted_information != {}:
        feedback2 = create_and_filtering_feedback(dataset, deleted_information, SFB_model, SFB_tokenizer, device)
        if '' not in feedback2:
            delete_success = True
        delete_feedback = ' and '.join(feedback2)
    delete_feedback = delete_feedback.replace('i need', 'i don\'t need').replace('i want', 'i don\'t want').replace('i\'m', 'i\'m not')\
    .replace('i am', 'i am not').replace('i would', 'i would not').replace('i\'d', 'i\'d not')\
    .replace('i\'ll', 'i\'ll not').replace('i will', 'i will not').replace('i don\'t care', 'i am not don\'t care')

    feedback = ''
    if add_success and not delete_success:
        feedback = random.sample(introducer_list[data_type], 1)[0] + ' , ' + add_feedback
    if not add_success and delete_success:
        feedback = random.sample(introducer_list[data_type], 1)[0] + ' , ' + delete_feedback
    if add_success and delete_success:
        feedback = random.sample(introducer_list[data_type], 1)[0] + ' , ' + add_feedback + ' and ' + delete_feedback
    return feedback, add_success, delete_success

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

def read_data(path_name, config_path, dataset):
    print(("Reading all files from {}".format(path_name)))
    with open(config_path, "r", encoding='utf-8') as f:
        raw_config = json.load(f)
    slot_meta = raw_config['slots']
    data = []
    with open(path_name) as f:
        dials = json.load(f)
        for dial_dict in dials:
            # Reading data
            for ti, turn in enumerate(dial_dict["dialogue"]):
                if dataset == 'MultiWOZ_2.1':
                    if(turn["domain"] not in EXPERIMENT_DOMAINS):
                        continue # We skip turns that doesn't appear in EXPERIMENT_DOMAINS
                dialog_state = process_state(turn["belief_state"], slot_meta)
                state_list = [[s,v] for s,v in dialog_state.items()]
                data_detail = {
                            "dialogue_idx": dial_dict["dialogue_idx"],
                            "turn_idx": turn["turn_idx"],
                            "system":turn["system_transcript"], 
                            "user":turn["transcript"],
                            "turn_label":turn["turn_label"],
                            "belief_state":turn["belief_state"],
                            "state_list":state_list
                            }
                data.append(data_detail)
    return data

def gen_time_pair():
    time_formats = ["am","pm","standard"]
    time_format = np.random.choice(time_formats,1)[0]
    if(time_format=="am" or time_format=="pm"):
        hour = random.randint(1,11)
        leave_min = random.randint(10,29)
        arrive_min = leave_min + random.randint(10,30)
        leave_time = str(hour)+":"+str(leave_min)+" "+time_format
        arrive_time = str(hour)+":"+str(arrive_min)+" "+time_format
    else:
        hour = random.randint(13,23)
        leave_min = random.randint(10,29)
        arrive_min = leave_min + random.randint(10,30)
        leave_time = str(hour)+":"+str(leave_min)
        arrive_time = str(hour)+":"+str(arrive_min)
    return(leave_time,arrive_time)

def fix_commonsense(turn_label_dict):
    if(("taxi-arriveby" in turn_label_dict) and ("taxi-leaveat" in turn_label_dict)):
        leave_time,arrive_time = gen_time_pair()
        turn_label_dict["taxi-leaveat"] = leave_time
        turn_label_dict["taxi-arriveby"] = arrive_time
    if(("train-arriveby" in turn_label_dict) and ("train-leaveat" in turn_label_dict)):
        leave_time,arrive_time = gen_time_pair()
        turn_label_dict["taxi-leaveat"] = leave_time
        turn_label_dict["taxi-arriveby"] = arrive_time
    return turn_label_dict

def feedback_state_generator(turn, max_add, max_delete, slot_value_dict):

    state = turn["state_list"]
    random.shuffle(state)
    num_add = 0
    if len(state) > 0:
        shuffle_list = [i for i in range(1, min(max_add, len(state))+1)]
        random.shuffle(shuffle_list)
        num_add = shuffle_list[0]
    num_delete = 0
    avail = len(state) - num_add
    if avail > 0:
        shuffle_list2 = [i for i in range(1, min(max_delete,avail)+1)]
        random.shuffle(shuffle_list2)
        num_delete = shuffle_list2[0]
    add_slot_dict = {sv[0]:sv[1] for sv in state[:num_add]}
    delete_slot_dict = {sv[0]:sv[1] for sv in state[num_add:num_add+num_delete]}
    for slot in add_slot_dict.keys():
        value_list = slot_value_dict[slot]
        value = np.random.choice(value_list,1)[0]
        while value == add_slot_dict[slot]:
            value = np.random.choice(value_list,1)[0]
        add_slot_dict[slot] = value
    add_slot_dict = fix_commonsense(add_slot_dict)
    return add_slot_dict, delete_slot_dict

    # turn = update_turn(turn,turn_label_dict)
    # return turn

def main():
    args_seed = 42
    args_dataset = "MultiWOZ_2.1"
    args_model_name_or_path = model_file[args_dataset]

    args_gene_data_save_dir = "feedback_data/%s"%(args_dataset)
    if args_dataset == "WOZ_2.0":
        args_data_file = "data/WOZ_2.0/woz_train_en.json"
        args_data_config_file = "data/dataset_config/woz2.json"
        args_data_ontology_file = "data/dataset_config/woz2_ontology.json"
        args_max_add = 2
        args_max_delete = 1
    elif args_dataset == "MultiWOZ_2.1":
        args_data_file = "data/MultiWOZ_2.1/train_dials.json"
        args_data_config_file = "data/dataset_config/multiwoz21.json"
        args_data_ontology_file = "data/dataset_config/multiwoz21_ontology.json"
        args_max_add = 4
        args_max_delete = 2
    else:
        print('select dataset in WOZ_2.0 and MultiWOZ_2.1')
        exit()

    args_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args_n_gpu = torch.cuda.device_count()
    set_seed(args_seed, args_n_gpu)
    model = T5ForConditionalGeneration.from_pretrained(args_model_name_or_path)
    tokenizer = T5Tokenizer.from_pretrained(args_model_name_or_path)
    model.to(args_device)

    dataset = read_data(args_data_file, args_data_config_file, args_dataset)
    with open(args_data_ontology_file, "r", encoding='utf-8') as f:
        data_ontology = json.load(f)

    success_gen = 0
    add_success_gen = 0
    delete_success_gen = 0
    save_info = {}
    print(len(dataset), " data points in total")
    for idx in tqdm(range(len(dataset))):
        turn = dataset[idx]
        new_turn = copy.deepcopy(turn)
        dialogue_idx = new_turn["dialogue_idx"]
        turn_idx = new_turn["turn_idx"]
        add_slot_dict, delete_slot_dict = feedback_state_generator(new_turn, args_max_add, args_max_delete, data_ontology)
        feedback, add_success, delete_success = feedback_utter_generator(args_dataset, 'train', add_slot_dict, delete_slot_dict, 
                                                model, tokenizer, args_device)
        new_turn = update_turn(new_turn, add_slot_dict, delete_slot_dict, add_success, delete_success)
        if add_success or delete_success:
            success_gen += 1
            if add_success:
                add_success_gen += 1
            if delete_success:
                delete_success_gen += 1
            save_info[str(dialogue_idx) + '-' + str(turn_idx)] = {
                                                        "feedback": feedback,
                                                        "ori_turn_label":turn["turn_label"],
                                                        "ori_belief_state":turn["belief_state"],
                                                        "new_turn_label": new_turn["turn_label"],
                                                        "belief_state": new_turn["belief_state"]}
    save_info["success rate"] = success_gen / (idx + 1)
    save_info["add success rate"] = add_success_gen / (idx + 1)
    save_info["delete success rate"] = delete_success_gen / (idx + 1)
    print("success generation rate: ", success_gen / (idx + 1))
    print("success add generation rate: ", add_success_gen / (idx + 1))
    print("success delete generation rate: ", delete_success_gen / (idx + 1))

    saved_file_name = "add[%s]_delete[%s]_seed[%s].json"%(args_max_add, args_max_delete, args_seed)

    if not os.path.exists(args_gene_data_save_dir):
        os.makedirs(args_gene_data_save_dir)

    with open(os.path.join(args_gene_data_save_dir, saved_file_name), "w") as f:
        json.dump(save_info, f, indent=4)

if __name__ == "__main__":
    main()
