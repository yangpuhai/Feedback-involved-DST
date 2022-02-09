import json
import re, random
from tqdm import tqdm

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]
# Required for mapping slot names in dialogue_acts.json file
# to proper designations.

LABEL_MAPS = {} # Loaded from file

def normalize_time(text):
    text = re.sub("(\d{1})(a\.?m\.?|p\.?m\.?)", r"\1 \2", text) # am/pm without space
    text = re.sub("(^| )(\d{1,2}) (a\.?m\.?|p\.?m\.?)", r"\1\2:00 \3", text) # am/pm short to long form
    text = re.sub("(^| )(at|from|by|until|after) ?(\d{1,2}) ?(\d{2})([^0-9]|$)", r"\1\2 \3:\4\5", text) # Missing separator
    text = re.sub("(^| )(\d{2})[;.,](\d{2})", r"\1\2:\3", text) # Wrong separator
    text = re.sub("(^| )(at|from|by|until|after) ?(\d{1,2})([;., ]|$)", r"\1\2 \3:00\4", text) # normalize simple full hour time
    text = re.sub("(^| )(\d{1}:\d{2})", r"\g<1>0\2", text) # Add missing leading 0
    # Map 12 hour times to 24 hour times
    text = re.sub("(\d{2})(:\d{2}) ?p\.?m\.?", lambda x: str(int(x.groups()[0]) + 12 if int(x.groups()[0]) < 12 else int(x.groups()[0])) + x.groups()[1], text)
    text = re.sub("(^| )24:(\d{2})", r"\g<1>00:\2", text) # Correct times that use 24 as hour
    return text


def normalize_text(text):
    text = normalize_time(text)
    text = re.sub("n't", " not", text)
    text = re.sub("(^| )zero(-| )star([s.,? ]|$)", r"\g<1>0 star\3", text)
    text = re.sub("(^| )one(-| )star([s.,? ]|$)", r"\g<1>1 star\3", text)
    text = re.sub("(^| )two(-| )star([s.,? ]|$)", r"\g<1>2 star\3", text)
    text = re.sub("(^| )three(-| )star([s.,? ]|$)", r"\g<1>3 star\3", text)
    text = re.sub("(^| )four(-| )star([s.,? ]|$)", r"\g<1>4 star\3", text)
    text = re.sub("(^| )five(-| )star([s.,? ]|$)", r"\g<1>5 star\3", text)
    text = re.sub("archaelogy", "archaeology", text) # Systematic typo
    text = re.sub("guesthouse", "guest house", text) # Normalization
    text = re.sub("(^| )b ?& ?b([.,? ]|$)", r"\1bed and breakfast\2", text) # Normalization
    text = re.sub("bed & breakfast", "bed and breakfast", text) # Normalization
    return text


# This should only contain label normalizations. All other mappings should
# be defined in LABEL_MAPS.
def normalize_label(slot, value_label):
    # Normalization of empty slots
    # import pdb;
    # pdb.set_trace()
    if value_label == '' or value_label == "not mentioned":
        return "none"

    # Normalization of time slots
    if "leaveat" in slot or "arriveby" in slot or slot == 'restaurant-book time':
        return normalize_time(value_label)

    # Normalization
    if "type" in slot or "name" in slot or "destination" in slot or "departure" in slot:
        value_label = re.sub("guesthouse", "guest house", value_label)

    # Map to boolean slots
    if slot == 'hotel-parking' or slot == 'hotel-internet':
        if value_label == 'yes' or value_label == 'free':
            return "true"
        if value_label == "no":
            return "false"
    if slot == 'hotel-type':
        if value_label == "hotel":
            return "true"
        if value_label == "guest house":
            return "false"

    return value_label


def get_token_pos(tok_list, value_label, tokenizer):
    find_pos = []
    found = False
    label_list  = tokenizer.tokenize(value_label)
    len_label = len(label_list)
    for i in range(len(tok_list) + 1 - len_label):
        if tok_list[i:i + len_label] == label_list:
            find_pos.append((i, i + len_label)) # start, exclusive_end
            found = True
    return found, find_pos


def check_label_existence(value_label, usr_utt_tok, sys_utt_tok, hst_utt_tok, tokenizer, append_history):
    in_usr, usr_pos = get_token_pos(usr_utt_tok, value_label, tokenizer)
    in_sys, sys_pos = get_token_pos(sys_utt_tok, value_label, tokenizer)
    in_hst, hst_pos = False, []
    if append_history:
        in_hst, hst_pos = get_token_pos(hst_utt_tok, value_label, tokenizer)
    # If no hit even though there should be one, check for value label variants
    if not in_usr and not in_sys and not in_hst and value_label in LABEL_MAPS:
        for value_label_variant in LABEL_MAPS[value_label]:
            in_usr, usr_pos = get_token_pos(usr_utt_tok, value_label_variant, tokenizer)
            in_sys, sys_pos = get_token_pos(sys_utt_tok, value_label_variant, tokenizer)
            if append_history:
                in_hst, hst_pos = get_token_pos(hst_utt_tok, value_label_variant, tokenizer)
            if in_usr or in_sys or in_hst:
                value_label = value_label_variant
                break
    return value_label, in_usr, usr_pos, in_sys, sys_pos, in_hst, hst_pos

def get_turn_label(value_label, sys_utt_tok, usr_utt_tok, hst_utt_tok, tokenizer, append_history, slot_last_occurrence):
    sys_utt_tok_label = [0 for _ in sys_utt_tok]
    usr_utt_tok_label = [0 for _ in usr_utt_tok]
    hst_utt_tok_label = [0 for _ in hst_utt_tok]
    if value_label == 'none' or value_label == 'dontcare' or value_label == 'true' or value_label == 'false':
        class_type = value_label
    else:
        value_label, in_usr, usr_pos, in_sys, sys_pos, in_hst, hst_pos = check_label_existence(
            value_label, usr_utt_tok, sys_utt_tok, hst_utt_tok, tokenizer, append_history)
        if in_usr or in_sys or in_hst:
            class_type = 'copy_value'
            if slot_last_occurrence:
                if in_usr:
                    (s, e) = usr_pos[-1]
                    for i in range(s, e):
                        usr_utt_tok_label[i] = 1
                elif in_sys:
                    (s, e) = sys_pos[-1]
                    for i in range(s, e):
                        sys_utt_tok_label[i] = 1
                else:
                    (s, e) = hst_pos[-1]
                    for i in range(s, e):
                        hst_utt_tok_label[i] = 1
            else:
                for (s, e) in usr_pos:
                    for i in range(s, e):
                        usr_utt_tok_label[i] = 1
                for (s, e) in sys_pos:
                    for i in range(s, e):
                        sys_utt_tok_label[i] = 1
                for (s, e) in hst_pos:
                    for i in range(s, e):
                        hst_utt_tok_label[i] = 1
        else:
            class_type = 'unpointable'
    return sys_utt_tok_label, usr_utt_tok_label, hst_utt_tok_label, class_type

def process_state(state, slot_meta):
    result = {}
    for s_dict in state:
        if s_dict['act'] == 'inform':
            slot = s_dict['slots'][0][0]
            if slot not in slot_meta:
                continue
            value = s_dict['slots'][0][1]
            value = normalize_label(slot, value)
            result[slot] = value
    return result

def create_examples(input_file, feedback_data_path, train_feedback, slot_list, set_type, tokenizer,
                    label_maps={},
                    append_history=False,
                    exclude_unpointable=True):
    """Read a DST json file into a list of DSTExample."""

    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)
    
    if set_type == 'train':
        with open(feedback_data_path) as f2:
            feedback_set = json.load(f2)
        feedback_keys = [k for k,v in feedback_set.items() if 'rate' not in k]
        random.seed(42)
        feedback_keys = random.sample(feedback_keys, int(train_feedback*len(feedback_keys)))
        random.seed()
        # print(len(feedback_keys))
        feedback_set = {k:v for k,v in feedback_set.items() if k in feedback_keys}
        # print(len(feedback_set))

    global LABEL_MAPS
    LABEL_MAPS = label_maps

    examples = []
    count = 0
    for dial in input_data:
        hst_utt_tok = []
        turn_num = 0
        for turn in dial['dialogue']:
            turn_domain = turn["domain"]
            if turn_domain in EXPERIMENT_DOMAINS:
                turn_num += 1
        for turn in dial['dialogue']:
            turn_domain = turn["domain"]
            if turn_domain not in EXPERIMENT_DOMAINS:
                continue
            guid = '%s-%s-%s' % (set_type,
                           str(dial['dialogue_idx']).replace('.json','').strip(),
                           str(turn['turn_idx']))
            sys_utt_tok = tokenizer.tokenize(normalize_text(turn['system_transcript'].lower()))
            usr_utt_tok = tokenizer.tokenize(normalize_text(turn['transcript'].lower()))
            turn_label = [[s, normalize_label(s, v)] for s, v in turn['turn_label']]
            turn_dialog_state = process_state(turn['belief_state'], slot_list)
            gold_state = [str(k) + '-' + str(v) for k, v in turn_dialog_state.items()]
            gold_turn_label = {}
            sys_utt_tok_label_dict = {}
            usr_utt_tok_label_dict = {}
            hst_utt_tok_label_dict = {}
            class_type_dict = {}
            for slot in slot_list:
                value_label = 'none'
                for [s, v] in turn_label:
                    if s == slot:
                        value_label = v
                        break
                gold_turn_label[slot] = value_label
                sys_utt_tok_label, usr_utt_tok_label, hst_utt_tok_label, class_type = get_turn_label(
                    value_label, sys_utt_tok, usr_utt_tok, hst_utt_tok, tokenizer, append_history, slot_last_occurrence=True)
                sys_utt_tok_label_dict[slot] = sys_utt_tok_label
                usr_utt_tok_label_dict[slot] = usr_utt_tok_label
                class_type_dict[slot] = class_type
                hst_utt_tok_label_dict[slot] = hst_utt_tok_label
            if 'unpointable' not in class_type_dict.values() or not exclude_unpointable:
                examples.append({
                    'guid':guid,
                    'turn_num':turn_num,
                    'turn_domain':turn_domain,
                    'text_a':sys_utt_tok,
                    'text_b':usr_utt_tok,
                    'history':hst_utt_tok,
                    'text_a_label':sys_utt_tok_label_dict,
                    'text_b_label':usr_utt_tok_label_dict,
                    'history_label':hst_utt_tok_label_dict,
                    'class_label':class_type_dict,
                    'gold_turn_label':gold_turn_label,
                    'turn_dialog_state':turn_dialog_state,
                    'gold_state':gold_state}
                )
            
            if set_type == 'train':
                fid = str(dial['dialogue_idx']) + '-' + str(turn['turn_idx'])
                if fid in feedback_set:
                    turn['turn_label'] = feedback_set[fid]["new_turn_label"]
                    turn['belief_state'] = feedback_set[fid]["belief_state"]
                    new_usr_utt_tok = usr_utt_tok + tokenizer.tokenize(normalize_text(feedback_set[fid]["feedback"]))
                    turn_label = [[s, normalize_label(s, v)] for s, v in turn['turn_label']]
                    turn_dialog_state = process_state(turn['belief_state'], slot_list)
                    gold_state = [str(k) + '-' + str(v) for k, v in turn_dialog_state.items()]
                    gold_turn_label = {}
                    sys_utt_tok_label_dict = {}
                    usr_utt_tok_label_dict = {}
                    hst_utt_tok_label_dict = {}
                    class_type_dict = {}
                    for slot in slot_list:
                        value_label = 'none'
                        for [s, v] in turn_label:
                            if s == slot:
                                value_label = v
                                break
                        gold_turn_label[slot] = value_label
                        sys_utt_tok_label, usr_utt_tok_label, hst_utt_tok_label, class_type = get_turn_label(
                            value_label, sys_utt_tok, new_usr_utt_tok, hst_utt_tok, tokenizer, append_history, slot_last_occurrence=True)
                        sys_utt_tok_label_dict[slot] = sys_utt_tok_label
                        usr_utt_tok_label_dict[slot] = usr_utt_tok_label
                        class_type_dict[slot] = class_type
                        hst_utt_tok_label_dict[slot] = hst_utt_tok_label
                    if 'unpointable' not in class_type_dict.values() or not exclude_unpointable:
                        examples.append({
                            'guid':guid,
                            'turn_num':turn_num,
                            'turn_domain':turn_domain,
                            'text_a':sys_utt_tok,
                            'text_b':new_usr_utt_tok,
                            'history':hst_utt_tok,
                            'text_a_label':sys_utt_tok_label_dict,
                            'text_b_label':usr_utt_tok_label_dict,
                            'history_label':hst_utt_tok_label_dict,
                            'class_label':class_type_dict,
                            'gold_turn_label':gold_turn_label,
                            'turn_dialog_state':turn_dialog_state,
                            'gold_state':gold_state}
                        )
            if append_history:
                hst_utt_tok = sys_utt_tok + usr_utt_tok
    return examples

