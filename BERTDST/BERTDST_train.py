import os
os.environ['CUDA_VISIBLE_DEVICES']="4"

from model.BERTDST import BERTDST
from pytorch_transformers import BertTokenizer, AdamW, WarmupLinearSchedule, BertConfig
from BERTDST_evaluation import model_evaluation

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import argparse
import random
import os
import json
import time
from copy import deepcopy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def masked_cross_entropy_for_value(logits, target, pad_idx=0):
    mask = target.ne(pad_idx)
    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = torch.log(logits_flat)
    target_flat = target.view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size())
    losses = losses * mask.float()
    loss = losses.sum()
    mask_sum = mask.sum().float()
    if mask_sum != 0:
        loss = loss / mask_sum
    #loss = losses.sum() / (mask.sum().float())
    return loss


def main(args):
    def worker_init_fn(worker_id):
        np.random.seed(args.random_seed + worker_id)
    
    if args.dataset == 'WOZ_2.0':
        from utils.WOZ_data_utils import prepare_dataset, MultiWozDataset, postprocessing, state_equal, SLOT, OP
    if args.dataset == 'MultiWOZ_2.1':
        from utils.MultiWOZ_data_utils import prepare_dataset, MultiWozDataset, postprocessing, state_equal, OP, make_slot_meta
        SLOT = make_slot_meta(args.config_path)

    n_gpu = 0
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    rng = random.Random(args.random_seed)
    torch.manual_seed(args.random_seed)
    if n_gpu > 0:
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    slot_meta = SLOT
    op2id = OP
    print(op2id)
    tokenizer = BertTokenizer(args.vocab_path, do_lower_case=True)

    if args.dataset == 'WOZ_2.0':
        train_data_path = args.train_data_path
        dev_data_path = args.dev_data_path
        test_data_path = args.test_data_path
    elif args.dataset == 'MultiWOZ_2.1':
        train_data_path=[args.train_data_path, args.config_path]
        dev_data_path=[args.dev_data_path, args.config_path]
        test_data_path=[args.test_data_path, args.config_path]

    train_data_raw = prepare_dataset(data_path=train_data_path,
                                     feedback_data_path=args.feedback_data_path,
                                     train_feedback=args.train_feedback,
                                     tokenizer=tokenizer,
                                     slot_meta=slot_meta,
                                     max_seq_length=args.max_seq_length,
                                     data_type='train',
                                     append_history=args.append_history)

    train_data = MultiWozDataset(train_data_raw,
                                 tokenizer,
                                 slot_meta,
                                 args.max_seq_length,
                                 rng,
                                 args.word_dropout)
    print("# train examples %d" % len(train_data_raw))

    dev_data_raw = prepare_dataset(data_path=dev_data_path,
                                   feedback_data_path=args.feedback_data_path,
                                   train_feedback=args.train_feedback,
                                   tokenizer=tokenizer,
                                   slot_meta=slot_meta,
                                   max_seq_length=args.max_seq_length,
                                   data_type='dev',
                                   append_history=args.append_history)
    
    dev_data = MultiWozDataset(dev_data_raw,
                               tokenizer,
                               slot_meta,
                               args.max_seq_length,
                               rng,
                               0.0)
    print("# dev examples %d" % len(dev_data_raw))

    test_data_raw = prepare_dataset(data_path=test_data_path,
                                    feedback_data_path=args.feedback_data_path,
                                    train_feedback=args.train_feedback,
                                    tokenizer=tokenizer,
                                    slot_meta=slot_meta,
                                    max_seq_length=args.max_seq_length,
                                    data_type='test',
                                    append_history=args.append_history)
    print("# test examples %d" % len(test_data_raw))
    # exit()

    model_config = BertConfig.from_json_file(args.bert_config_path)
    model_config.dropout = args.dropout
    model_config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
    model_config.hidden_dropout_prob = args.hidden_dropout_prob
    model = BERTDST(model_config, len(op2id), len(slot_meta))

    ckpt = torch.load(args.bert_ckpt_path, map_location='cpu')
    ckpt1 = {k.replace('bert.', '').replace('gamma','weight').replace('beta','bias'): v for k, v in ckpt.items() if 'cls.' not in k}
    model.encoder.bert.load_state_dict(ckpt1)
    #model.encoder.bert.from_pretrained(args.bert_ckpt_path)

    model.to(device)

    num_train_steps = int(len(train_data_raw) / args.batch_size * args.n_epochs)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    enc_param_optimizer = list(model.encoder.named_parameters())
    enc_optimizer_grouped_parameters = [
        {'params': [p for n, p in enc_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in enc_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    enc_optimizer = AdamW(enc_optimizer_grouped_parameters, lr=args.enc_lr)
    enc_scheduler = WarmupLinearSchedule(enc_optimizer, int(num_train_steps * args.enc_warmup),
                                         t_total=num_train_steps)

    dec_param_optimizer = list(model.decoder.parameters())
    dec_optimizer = AdamW(dec_param_optimizer, lr=args.dec_lr)
    dec_scheduler = WarmupLinearSchedule(dec_optimizer, int(num_train_steps * args.dec_warmup),
                                         t_total=num_train_steps)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data,
                                  sampler=train_sampler,
                                  batch_size=args.batch_size,
                                  collate_fn=train_data.collate_fn,
                                  num_workers=args.num_workers,
                                  worker_init_fn=worker_init_fn)
    
    dev_sampler = RandomSampler(dev_data)
    dev_dataloader = DataLoader(dev_data,
                                sampler=dev_sampler,
                                batch_size=args.batch_size,
                                collate_fn=train_data.collate_fn,
                                num_workers=args.num_workers,
                                worker_init_fn=worker_init_fn)

    loss_fnc = nn.CrossEntropyLoss()
    best_score = {'epoch': 0, 'mean_loss': 0}
    total_step = 0
    for epoch in range(1, args.n_epochs+1):
        batch_loss = []
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = [b.to(device) if not isinstance(b, int) else b for b in batch]
            input_ids, input_mask, segment_ids, op_ids, span_ids = batch

            state_scores, span_scores = model(input_ids=input_ids,
                                              token_type_ids=segment_ids,
                                              attention_mask=input_mask)

            loss_state = loss_fnc(state_scores.contiguous().view(-1, len(op2id)), op_ids.contiguous().view(-1))
            try:
                loss_span = masked_cross_entropy_for_value(span_scores.contiguous(), span_ids.contiguous(), 0)
            except Exception as e:
                print(e)
            loss = loss_state * 0.8 + loss_span * 0.2
            batch_loss.append(loss.item())

            loss.backward()
            enc_optimizer.step()
            enc_scheduler.step()
            dec_optimizer.step()
            dec_scheduler.step()
            model.zero_grad()

            total_step += 1

            if step % 100 == 0:
                print("[%d/%d] [%d/%d] mean_loss : %.3f, state_loss : %.3f, span_loss : %.3f" \
                          % (epoch, args.n_epochs, step,
                             len(train_dataloader), np.mean(batch_loss),
                             loss_state.item(), loss_span.item()))
                batch_loss = []

        if epoch % args.eval_epoch == 0:
            print('total_step: ',total_step)
            dev_batch_loss = []
            model.eval()
            for dev_batch in dev_dataloader:
                dev_batch = [b.to(device) if not isinstance(b, int) else b for b in dev_batch]
                input_ids, input_mask, segment_ids, op_ids, span_ids = dev_batch
                state_scores, span_scores = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
                dev_loss_state = loss_fnc(state_scores.contiguous().view(-1, len(op2id)), op_ids.contiguous().view(-1))
                dev_loss_span = masked_cross_entropy_for_value(span_scores.contiguous(), span_ids.contiguous(), 0)
                dev_loss = dev_loss_state * 0.8 + dev_loss_span * 0.2
                dev_batch_loss.append(dev_loss.item())
            dev_mean_loss = np.mean(dev_batch_loss)
            if epoch == 1 or dev_mean_loss < best_score['mean_loss']:
                best_score['epoch'] = epoch
                best_score['mean_loss'] = dev_mean_loss
                save_path = os.path.join(args.save_dir, 'model_best_epoch[%s]_maxlen[%s]_feedback[%s]_seed[%s].bin'% (str(args.n_epochs), str(args.max_seq_length), args.train_feedback, args.random_seed))
                torch.save(model.state_dict(), save_path)
            print("Best Score : ", best_score)
            print("\n")

            if best_score['epoch'] + args.patience < epoch:
                print("out of patience...")
                break

    print("Test using best model...")
    best_epoch = best_score['epoch']
    ckpt_path = os.path.join(args.save_dir, 'model_best_epoch[%s]_maxlen[%s]_feedback[%s]_seed[%s].bin'% (str(args.n_epochs), str(args.max_seq_length), args.train_feedback, args.random_seed))
    model = BERTDST(model_config, len(op2id), len(slot_meta))
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt)
    model.to(device)

    model_evaluation(args, postprocessing, state_equal, OP, model, test_data_raw, tokenizer, slot_meta, best_epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--dataset", default='WOZ_2.0', type=str)
    parser.add_argument("--vocab_path", default='bert-base-uncased/vocab.txt', type=str)
    parser.add_argument("--bert_config_path", default='bert-base-uncased/config.json', type=str)
    parser.add_argument("--bert_ckpt_path", default='./bert-base-uncased/pytorch_model.bin', type=str)


    parser.add_argument("--random_seed", default=42, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--enc_warmup", default=0.1, type=float)
    parser.add_argument("--dec_warmup", default=0.1, type=float)
    parser.add_argument("--enc_lr", default=4e-5, type=float)
    parser.add_argument("--dec_lr", default=1e-4, type=float)
    parser.add_argument("--n_epochs", default=200, type=int)
    parser.add_argument("--eval_epoch", default=1, type=int)
    parser.add_argument("--patience", default=6, type=int)
    parser.add_argument("--append_history", default=True)
    parser.add_argument("--train_feedback", default=1.0, type=float)

    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float)
    parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float)
    parser.add_argument("--word_dropout", default=0.1, type=float)

    parser.add_argument("--max_seq_length", default=150, type=int)
    parser.add_argument("--msg", default=None, type=str)

    args = parser.parse_args()

    if args.dataset == 'WOZ_2.0':
        data_root = 'data/WOZ_2.0'
        config_root = 'data/dataset_config'
        feedback_data_root = 'feedback_data/WOZ_2.0'
        args.train_data_path = os.path.join(data_root, 'woz_train_en.json')
        args.dev_data_path = os.path.join(data_root, 'woz_validate_en.json')
        args.test_data_path = os.path.join(data_root, 'woz_test_en.json')
        args.config_path = os.path.join(config_root, 'woz2.json')
        args.feedback_data_path = os.path.join(feedback_data_root, 'add[2]_delete[1]_seed[42].json')
        args.save_dir = 'outputs/BERTDST/WOZ_outputs'
    elif args.dataset == 'MultiWOZ_2.1':
        data_root = 'data/MultiWOZ_2.1'
        config_root = 'data/dataset_config'
        feedback_data_root = 'feedback_data/MultiWOZ_2.1'
        args.train_data_path = os.path.join(data_root, 'train_dials.json')
        args.dev_data_path = os.path.join(data_root, 'dev_dials.json')
        args.test_data_path = os.path.join(data_root, 'test_dials.json')
        args.config_path = os.path.join(config_root, 'multiwoz21.json')
        args.feedback_data_path = os.path.join(feedback_data_root, 'add[4]_delete[2]_seed[42].json')
        args.save_dir = 'outputs/BERTDST/MultiWOZ_outputs'
    else:
        print('select dataset in WOZ_2.0 and MultiWOZ_2.1')
        exit()
    print('pytorch version: ', torch.__version__)
    print(args)
    main(args)