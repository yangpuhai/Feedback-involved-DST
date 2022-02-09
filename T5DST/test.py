
import os, random
os.environ['CUDA_VISIBLE_DEVICES']="1"

import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from transformers import (AdamW, T5Tokenizer, BartTokenizer, BartForConditionalGeneration, T5ForConditionalGeneration, WEIGHTS_NAME,CONFIG_NAME)
from data_loader import prepare_data
from evaluate import evaluate_metrics
import json
from tqdm import tqdm
from T5 import DST_Seq2Seq

def test(args, *more):
    args = vars(args)
    args["model_name"] = args["model_checkpoint"]+ "_slotlang_" +str(args["slot_lang"]) + "_lr_" +str(args["lr"]) + "_epoch_" + str(args["n_epochs"]) + "_seed_" + str(args["seed"]) + "_feedback_" + str(args["train_feedback"])
    save_path = os.path.join(args["saving_dir"], args["dataset"], args["model_name"])

    model = T5ForConditionalGeneration.from_pretrained(save_path)
    tokenizer = T5Tokenizer.from_pretrained(save_path)

    task = DST_Seq2Seq(args, tokenizer, model)

    data_test, test_loader, ALL_SLOTS = prepare_data(args, task.tokenizer)

    print("test start...")
    #evaluate model
    _ = evaluate_model(args, task.tokenizer, task.model, test_loader, save_path, ALL_SLOTS)

def evaluate_model(args, tokenizer, model, test_loader, save_path, ALL_SLOTS):
    save_path = os.path.join(save_path,"results")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    predictions = {}
    # to gpu
    # gpu = args["GPU"][0]
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()

    slot_logger = {slot_name:[0,0,0] for slot_name in ALL_SLOTS}

    for batch in tqdm(test_loader):
        dst_outputs = model.generate(input_ids=batch["encoder_input"].to(device),
                                attention_mask=batch["attention_mask"].to(device),
                                eos_token_id=tokenizer.eos_token_id,
                                max_length=200,
                                )

        value_batch = tokenizer.batch_decode(dst_outputs, skip_special_tokens=True)

        for idx, value in enumerate(value_batch):
            dial_id = batch["ID"][idx]
            if dial_id not in predictions:
                predictions[dial_id] = {}
                predictions[dial_id]["domain"] = batch["domains"][idx][0]
                predictions[dial_id]["turns"] = {}
            if batch["turn_id"][idx] not in predictions[dial_id]["turns"]:
                predictions[dial_id]["turns"][batch["turn_id"][idx]] = {"turn_belief":batch["turn_belief"][idx], "pred_belief":[]}

            if value!="none":
                predictions[dial_id]["turns"][batch["turn_id"][idx]]["pred_belief"].append(str(batch["slot_text"][idx])+'-'+str(value))

            # analyze slot acc:
            if str(value)==str(batch["value_text"][idx]):
                slot_logger[str(batch["slot_text"][idx])][1]+=1 # hit
            slot_logger[str(batch["slot_text"][idx])][0]+=1 # total

    for slot_log in slot_logger.values():
        slot_log[2] = slot_log[1]/slot_log[0]

    with open(os.path.join(save_path, f"slot_acc.json"), 'w') as f:
        json.dump(slot_logger,f, indent=4)

    with open(os.path.join(save_path, f"prediction.json"), 'w') as f:
        json.dump(predictions,f, indent=4)

    joint_acc_score, F1_score, turn_acc_score = evaluate_metrics(predictions, ALL_SLOTS)

    evaluation_metrics = {"Joint Acc":joint_acc_score, "Turn Acc":turn_acc_score, "Joint F1":F1_score}
    print(f"result:",evaluation_metrics)

    with open(os.path.join(save_path, f"result.json"), 'w') as f:
        json.dump(evaluation_metrics,f, indent=4)

    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MultiWOZ_2.1", help="WOZ_2.0 or MultiWOZ_2.1")
    parser.add_argument("--model_checkpoint", type=str, default="t5-small", help="Path, url or short name of the model")
    parser.add_argument("--saving_dir", type=str, default="save", help="Path for saving")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--dev_batch_size", type=int, default=8, help="Batch size for validation")
    parser.add_argument("--test_batch_size", type=int, default=8, help="Batch size for test")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=557, help="Random seed")
    parser.add_argument("--GPU", type=int, default=1, help="number of gpu to use")
    parser.add_argument("--slot_lang", type=str, default="slottype", help="use 'none', 'human', 'naive', 'value', 'question', 'slottype' slot description")
    parser.add_argument("--train_feedback", default=1.0, type=float)
    args = parser.parse_args()

    if args.dataset == 'WOZ_2.0':
        data_root = 'data/WOZ_2.0'
        config_root = 'data/dataset_config'
        args.train_data_path = os.path.join(data_root, 'woz_train_en.json')
        args.dev_data_path = os.path.join(data_root, 'woz_validate_en.json')
        args.test_data_path = os.path.join(data_root, 'woz_test_en.json')
        args.config_path = os.path.join(config_root, 'woz2.json')
        args.description_path = os.path.join('utils', 'slot_description_WOZ.json')
    elif args.dataset == 'MultiWOZ_2.1':
        data_root = 'data/MultiWOZ_2.1'
        config_root = 'data/dataset_config'
        args.train_data_path = os.path.join(data_root, 'train_dials.json')
        args.dev_data_path = os.path.join(data_root, 'dev_dials.json')
        args.test_data_path = os.path.join(data_root, 'test_dials.json')
        args.config_path = os.path.join(config_root, 'multiwoz21.json')
        args.description_path = os.path.join('utils', 'slot_description_MWOZ.json')
    else:
        print('select dataset in WOZ_2.0 and MultiWOZ_2.1')
        exit()
    
    args.mode = 'test'
    print(args)
    test(args)
