import json
import re,random

# label_maps in config file woz2.json
SEMANTIC_DICT = {
  'center': ['centre', 'downtown', 'central', 'down town', 'middle'],
  'south': ['southern', 'southside'],
  'north': ['northern', 'uptown', 'northside'],
  'west': ['western', 'westside'],
  'east': ['eastern', 'eastside'],
  'east side': ['eastern', 'eastside'],

  'cheap': ['low price', 'inexpensive', 'cheaper', 'low priced', 'affordable',
            'nothing too expensive', 'without costing a fortune', 'cheapest',
            'good deals', 'low prices', 'afford', 'on a budget', 'fair prices',
            'less expensive', 'cheapeast', 'not cost an arm and a leg'],
  'moderate': ['moderately', 'medium priced', 'medium price', 'fair price',
               'fair prices', 'reasonable', 'reasonably priced', 'mid price',
               'fairly priced', 'not outrageous','not too expensive',
               'on a budget', 'mid range', 'reasonable priced', 'less expensive',
               'not too pricey', 'nothing too expensive', 'nothing cheap',
               'not overpriced', 'medium', 'inexpensive'],
  'expensive': ['high priced', 'high end', 'high class', 'high quality',
                'fancy', 'upscale', 'nice', 'fine dining', 'expensively priced'],

  'afghan': ['afghanistan'],
  'african': ['africa'],
  'asian oriental': ['asian', 'oriental'],
  'australasian': ['australian asian', 'austral asian'],
  'australian': ['aussie'],
  'barbeque': ['barbecue', 'bbq'],
  'basque': ['bask'],
  'belgian': ['belgium'],
  'british': ['cotto'],
  'canapes': ['canopy', 'canape', 'canap'],
  'catalan': ['catalonian'],
  'corsican': ['corsica'],
  'crossover': ['cross over', 'over'],
  'gastropub': ['gastro pub', 'gastro', 'gastropubs'],
  'hungarian': ['goulash'],
  'indian': ['india', 'indians', 'nirala'],
  'international': ['all types of food'],
  'italian': ['prezzo'],
  'jamaican': ['jamaica'],
  'japanese': ['sushi', 'beni hana'],
  'korean': ['korea'],
  'lebanese': ['lebanse'],
  'north american': ['american', 'hamburger'],
  'portuguese': ['portugese'],
  'seafood': ['sea food', 'shellfish', 'fish'],
  'singaporean': ['singapore'],
  'steakhouse': ['steak house', 'steak'],
  'thai': ['thailand', 'bangkok'],
  'traditional': ['old fashioned', 'plain'],
  'turkish': ['turkey'],
  'unusual': ['unique and strange'],
  'venetian': ['vanessa'],
  'vietnamese': ['vietnam', 'thanh binh'],
                  }

FIX = {'centre': 'center', 'areas': 'area', 'phone number': 'number'}


def get_token_pos(tok_list, label, tokenizer):
  find_pos = []
  found = False
  label_list  = tokenizer.tokenize(label)
  len_label = len(label_list)
  for i in range(len(tok_list) + 1 - len_label):
    if tok_list[i:i+len_label] == label_list:
      find_pos.append((i,i+len_label))  # start, exclusive_end
      found = True
  return found, find_pos


def check_label_existence(label, usr_utt_tok, sys_utt_tok, tokenizer):
  in_usr, usr_pos = get_token_pos(usr_utt_tok, label, tokenizer)
  in_sys, sys_pos = get_token_pos(sys_utt_tok, label, tokenizer)

  if not in_usr and not in_sys and label in SEMANTIC_DICT:
    for tmp_label in SEMANTIC_DICT[label]:
      in_usr, usr_pos = get_token_pos(usr_utt_tok, tmp_label, tokenizer)
      in_sys, sys_pos = get_token_pos(sys_utt_tok, tmp_label, tokenizer)
      if in_usr or in_sys:
        label = tmp_label
        break
  return label, in_usr, usr_pos, in_sys, sys_pos


def get_turn_label(label, sys_utt_tok, usr_utt_tok, tokenizer, slot_last_occurrence):
  sys_utt_tok_label = [0 for _ in sys_utt_tok]
  usr_utt_tok_label = [0 for _ in usr_utt_tok]
  if label == 'none' or label == 'dontcare':
    class_type = label
  else:
    label, in_usr, usr_pos, in_sys, sys_pos = check_label_existence(label, usr_utt_tok, sys_utt_tok, tokenizer)
    if in_usr or in_sys:
      class_type = 'copy_value'
      if slot_last_occurrence:
        if in_usr:
          (s, e) = usr_pos[-1]
          for i in range(s, e):
            usr_utt_tok_label[i] = 1
        else:
          (s, e) = sys_pos[-1]
          for i in range(s, e):
            sys_utt_tok_label[i] = 1
      else:
        for (s, e) in usr_pos:
          for i in range(s, e):
            usr_utt_tok_label[i] = 1
        for (s, e) in sys_pos:
          for i in range(s, e):
            sys_utt_tok_label[i] = 1
    else:
      class_type = 'unpointable'
  return sys_utt_tok_label, usr_utt_tok_label, class_type

def process_state(state, slot_meta):
    result = {}
    for s_dict in state:
        if s_dict['act'] == 'inform':
            slot = s_dict['slots'][0][0]
            if slot not in slot_meta:
                continue
            value = s_dict['slots'][0][1]
            slot = FIX.get(slot.strip(), slot.strip())
            value = FIX.get(value.strip(), value.strip())
            result[slot] = value
    return result

def create_examples(dialog_filename, feedback_data_path, train_feedback, slot_list, set_type, tokenizer, use_asr_hyp=0,
                    exclude_unpointable=True):
  examples = []
  with open(dialog_filename) as f:
    dst_set = json.load(f)
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

  for dial in dst_set:
    turn_num = 0
    for turn in dial['dialogue']:
      turn_num += 1
    for turn in dial['dialogue']:
      guid = '%s-%s-%s' % (set_type,
                           str(dial['dialogue_idx']),
                           str(turn['turn_idx']))

      sys_utt_tok = tokenizer.tokenize(turn['system_transcript'])

      if use_asr_hyp == 0:
        usr_utt_tok = tokenizer.tokenize(turn['transcript'])
      else:
        for asr_hyp, _ in turn['asr'][:use_asr_hyp]:
          usr_utt_tok = tokenizer.tokenize(asr_hyp)
      
      turn_label = [[FIX.get(s.strip(), s.strip()), FIX.get(v.strip(), v.strip())] for s, v in turn['turn_label']]
      gold_turn_label = {}
      turn_dialog_state = process_state(turn['belief_state'], slot_list)
      gold_state = [str(k) + '-' + str(v) for k, v in turn_dialog_state.items()]
      sys_utt_tok_label_dict = {}
      usr_utt_tok_label_dict = {}
      class_type_dict = {}
      for slot in slot_list:
        label = 'none'
        for [s, v] in turn_label:
          if s == slot:
            label = v
            break
        gold_turn_label[slot] = label
        sys_utt_tok_label, usr_utt_tok_label, class_type = get_turn_label(
          label, sys_utt_tok, usr_utt_tok, tokenizer,
          slot_last_occurrence=True)
        sys_utt_tok_label_dict[slot] = sys_utt_tok_label
        usr_utt_tok_label_dict[slot] = usr_utt_tok_label
        class_type_dict[slot] = class_type
      # print('guid',guid)
      # print('sys_utt_tok',sys_utt_tok)
      # print('usr_utt_tok',usr_utt_tok)
      # print('sys_utt_tok_label_dict',sys_utt_tok_label_dict)
      # print('usr_utt_tok_label_dict',usr_utt_tok_label_dict)
      # print('class_type_dict',class_type_dict)
      if 'unpointable' not in class_type_dict.values() or not exclude_unpointable:
        examples.append({
          'guid':guid,
          'turn_num':turn_num,
          'text_a':sys_utt_tok,
          'text_b':usr_utt_tok,
          'text_a_label':sys_utt_tok_label_dict,
          'text_b_label':usr_utt_tok_label_dict,
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
          usr_utt_tok = usr_utt_tok + tokenizer.tokenize(feedback_set[fid]["feedback"])
          turn_label = [[FIX.get(s.strip(), s.strip()), FIX.get(v.strip(), v.strip())] for s, v in turn['turn_label']]
          gold_turn_label = {}
          turn_dialog_state = process_state(turn['belief_state'], slot_list)
          gold_state = [str(k) + '-' + str(v) for k, v in turn_dialog_state.items()]
          sys_utt_tok_label_dict = {}
          usr_utt_tok_label_dict = {}
          class_type_dict = {}
          for slot in slot_list:
            label = 'none'
            for [s, v] in turn_label:
              if s == slot:
                label = v
                break
            gold_turn_label[slot] = label
            sys_utt_tok_label, usr_utt_tok_label, class_type = get_turn_label(
              label, sys_utt_tok, usr_utt_tok, tokenizer,
              slot_last_occurrence=True)
            sys_utt_tok_label_dict[slot] = sys_utt_tok_label
            usr_utt_tok_label_dict[slot] = usr_utt_tok_label
            class_type_dict[slot] = class_type
          if 'unpointable' not in class_type_dict.values() or not exclude_unpointable:
              examples.append({
                'guid':guid,
                'turn_num':turn_num,
                'text_a':sys_utt_tok,
                'text_b':usr_utt_tok,
                'text_a_label':sys_utt_tok_label_dict,
                'text_b_label':usr_utt_tok_label_dict,
                'class_label':class_type_dict,
                'gold_turn_label':gold_turn_label,
                'turn_dialog_state':turn_dialog_state,
                'gold_state':gold_state}
                )
      
  return examples

