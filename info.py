import random
from Feedback_Simulation.data_loader import linear_turn_label_MWOZ, linear_turn_label_WOZ

model_file = {
    'MultiWOZ_2.1': 'Feedback_Simulation/save/MultiWOZ_2.2/t5-small_lr_0.0005_epoch_20_seed_42',
    'WOZ_2.0': 'Feedback_Simulation/save/WOZ_2.0/t5-small_lr_0.0005_epoch_20_seed_42',
}

introducer_list = {'train':['you made a mistake', 'you are wrong', 'the system is wrong', 'you misunderstand what i mean', 'there are bugs in the system'],
                   'dev':['you made a mistake', 'you are wrong', 'the system is wrong', 'you misunderstand', 'there\'s something wrong with the system'],
                   'test':['you made a mistake', 'the system is wrong', 'you got it wrong', 'something went wrong', 'i am afraid you are wrong']}

def create_feedback(dataset, information, SFB_model, SFB_tokenizer, device):
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

        str1 = value_batch[0]
        for s1, s2 in zip(value_batch, value_batch2):
            if s2 == domain_slot_value_str and '?' not in s1:
                str1 = s1
                break
        # print(str1)
        feedback.append(str1)
    return ' and '.join(feedback)

def simulated_negative_feedback(dataset, data_type, added_information, deleted_information, SFB_model, SFB_tokenizer, device):
    feedback1 = ''
    if added_information != {}:
        feedback1 = create_feedback(dataset, added_information, SFB_model, SFB_tokenizer, device)
        # print(feedback1)
    feedback2 = ''
    if deleted_information != {}:
        feedback2 = create_feedback(dataset, deleted_information, SFB_model, SFB_tokenizer, device)
        # print(feedback2)
    feedback2 = feedback2.replace('need', 'don\'t need').replace('want', 'don\'t want').replace('i\'m', 'i\'m not')\
    .replace('i am', 'i am not').replace('i would', 'i would not').replace('i\'d', 'i\'d not')\
    .replace('i\'ll', 'i\'ll not').replace('i don\'t care', 'i am not don\'t care')

    if deleted_information == {}:
        return random.sample(introducer_list[data_type], 1)[0] + ' , ' + feedback1
    if added_information == {}:
        return random.sample(introducer_list[data_type], 1)[0] + ' , ' + feedback2
    return random.sample(introducer_list[data_type], 1)[0] + ' , ' + feedback1 + ' and ' + feedback2