import os
os.environ['CUDA_VISIBLE_DEVICES']="0"

from model.SOMDST import SOMDST
from pytorch_transformers import BertTokenizer, AdamW, WarmupLinearSchedule, BertConfig
from SOMDST_evaluation import model_evaluation

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import argparse
import random
import os
import json
import time


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
    # loss = losses.sum() / (mask.sum().float())
    return loss


def main(args):
    def worker_init_fn(worker_id):
        np.random.seed(args.random_seed + worker_id)

    if args.dataset == 'WOZ_2.0':
        from SOMDST_utils.WOZ_data_utils import prepare_dataset, MultiWozDataset, postprocessing, state_equal, SLOT, domain2id, OP_SET
    if args.dataset == 'MultiWOZ_2.1':
        from SOMDST_utils.MultiWOZ_data_utils import prepare_dataset, MultiWozDataset, postprocessing, state_equal, domain2id, OP_SET, make_slot_meta
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
    op2id = OP_SET[args.op_code]
    print(op2id)
    print(slot_meta)
    tokenizer = BertTokenizer(args.vocab_path, do_lower_case=True)

    train_data_raw = prepare_dataset(data_path=args.train_data_path,
                                     feedback_data_path=args.feedback_data_path,
                                     train_feedback=args.train_feedback,
                                     tokenizer=tokenizer,
                                     slot_meta=slot_meta,
                                     n_history=args.n_history,
                                     max_seq_length=args.max_seq_length,
                                     op_code=args.op_code,
                                     data_type='train')

    train_data = MultiWozDataset(train_data_raw,
                                 tokenizer,
                                 slot_meta,
                                 args.max_seq_length,
                                 rng,
                                 args.word_dropout,
                                 args.shuffle_state,
                                 args.shuffle_p)
    print("# train examples %d" % len(train_data_raw))
    #exit()
    dev_data_raw = prepare_dataset(data_path=args.dev_data_path,
                                   feedback_data_path=args.feedback_data_path,
                                   train_feedback=args.train_feedback,
                                   tokenizer=tokenizer,
                                   slot_meta=slot_meta,
                                   n_history=args.n_history,
                                   max_seq_length=args.max_seq_length,
                                   op_code=args.op_code,
                                   data_type='dev')

    print("# dev examples %d" % len(dev_data_raw))

    test_data_raw = prepare_dataset(data_path=args.test_data_path,
                                    feedback_data_path=args.feedback_data_path,
                                    train_feedback=args.train_feedback,
                                    tokenizer=tokenizer,
                                    slot_meta=slot_meta,
                                    n_history=args.n_history,
                                    max_seq_length=args.max_seq_length,
                                    op_code=args.op_code,
                                    data_type='test')
    print("# test examples %d" % len(test_data_raw))

    model_config = BertConfig.from_json_file(args.bert_config_path)
    model_config.dropout = args.dropout
    model_config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
    model_config.hidden_dropout_prob = args.hidden_dropout_prob
    model = SOMDST(model_config, len(op2id), len(domain2id), op2id['update'], args.exclude_domain)

    ckpt = torch.load(args.bert_ckpt_path, map_location='cpu')
    ckpt1 = {k.replace('bert.', '').replace('gamma','weight').replace('beta','bias'): v for k, v in ckpt.items() if 'cls.' not in k}
    model.encoder.bert.load_state_dict(ckpt1)
    #model.encoder.bert.from_pretrained(args.bert_ckpt_path)


    # re-initialize added special tokens ([SLOT], [NULL], [EOS])
    model.encoder.bert.embeddings.word_embeddings.weight.data[1].normal_(mean=0.0, std=0.02)
    model.encoder.bert.embeddings.word_embeddings.weight.data[2].normal_(mean=0.0, std=0.02)
    model.encoder.bert.embeddings.word_embeddings.weight.data[3].normal_(mean=0.0, std=0.02)
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

    loss_fnc = nn.CrossEntropyLoss()
    # best_score = {'epoch': 0, 'mean_loss': 0}
    best_score = {'epoch': 0, 'joint_acc': 0, 'slot_acc': 0, 'slot_f1': 0}
    total_step = 0
    for epoch in range(1, args.n_epochs+1):
        batch_loss = []
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = [b.to(device) if not isinstance(b, int) else b for b in batch]
            input_ids, input_mask, segment_ids, state_position_ids, op_ids,\
            domain_ids, gen_ids, max_value, max_update = batch

            if rng.random() < args.decoder_teacher_forcing:  # teacher forcing
                teacher = gen_ids
            else:
                teacher = None

            domain_scores, state_scores, gen_scores = model(input_ids=input_ids,
                                                            token_type_ids=segment_ids,
                                                            state_positions=state_position_ids,
                                                            attention_mask=input_mask,
                                                            max_value=max_value,
                                                            op_ids=op_ids,
                                                            max_update=max_update,
                                                            teacher=teacher)

            loss_s = loss_fnc(state_scores.view(-1, len(op2id)), op_ids.view(-1))
            loss_g = masked_cross_entropy_for_value(gen_scores.contiguous(),
                                                    gen_ids.contiguous(),
                                                    tokenizer.vocab['[PAD]'])
            loss = loss_s + loss_g
            if args.exclude_domain is not True:
                loss_d = loss_fnc(domain_scores.view(-1, len(domain2id)), domain_ids.view(-1))
                loss = loss + loss_d
            batch_loss.append(loss.item())

            loss.backward()
            enc_optimizer.step()
            enc_scheduler.step()
            dec_optimizer.step()
            dec_scheduler.step()
            model.zero_grad()

            total_step += 1

            if step % 100 == 0:
                if args.exclude_domain is not True:
                    print("[%d/%d] [%d/%d] mean_loss : %.3f, state_loss : %.3f, gen_loss : %.3f, dom_loss : %.3f" \
                          % (epoch, args.n_epochs, step,
                             len(train_dataloader), np.mean(batch_loss),
                             loss_s.item(), loss_g.item(), loss_d.item()))
                else:
                    print("[%d/%d] [%d/%d] mean_loss : %.3f, state_loss : %.3f, gen_loss : %.3f" \
                          % (epoch, args.n_epochs, step,
                             len(train_dataloader), np.mean(batch_loss),
                             loss_s.item(), loss_g.item()))
                batch_loss = []

        if epoch % args.eval_epoch == 0:
            print('total_step: ',total_step)
            print('evaluate...')
            eval_res = model_evaluation(args, postprocessing, state_equal, domain2id, OP_SET, model, dev_data_raw, tokenizer, slot_meta, epoch, args.op_code)
            if eval_res['joint_acc'] > best_score['joint_acc']:
                best_score = eval_res
                model_to_save = model.module if hasattr(model, 'module') else model
                save_path = os.path.join(args.save_dir, 'model_best_feedback[%s]_seed[%s].bin'% (args.train_feedback, args.random_seed))
                torch.save(model_to_save.state_dict(), save_path)
            print("Best Score : ", best_score)
            print("\n")

            if best_score['epoch'] + args.patience < epoch:
                print("out of patience...")
                break

    print("Test using best model...")
    best_epoch = best_score['epoch']
    ckpt_path = os.path.join(args.save_dir, 'model_best_feedback[%s]_seed[%s].bin'% (args.train_feedback, args.random_seed))
    model = SOMDST(model_config, len(op2id), len(domain2id), op2id['update'], args.exclude_domain)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt)
    model.to(device)

    model_evaluation(args, postprocessing, state_equal, domain2id, OP_SET, model, test_data_raw, tokenizer, slot_meta, best_epoch, args.op_code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--dataset", default='MultiWOZ_2.1', type=str)
    parser.add_argument("--vocab_path", default='bert-base-uncased/vocab.txt', type=str)
    parser.add_argument("--bert_config_path", default='bert-base-uncased/config.json', type=str)
    parser.add_argument("--bert_ckpt_path", default='./bert-base-uncased/pytorch_model.bin', type=str)

    parser.add_argument("--random_seed", default=42, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--enc_warmup", default=0.1, type=float)
    parser.add_argument("--dec_warmup", default=0.1, type=float)
    parser.add_argument("--enc_lr", default=4e-5, type=float)
    parser.add_argument("--dec_lr", default=1e-4, type=float)
    parser.add_argument("--n_epochs", default=30, type=int)
    parser.add_argument("--eval_epoch", default=1, type=int)
    parser.add_argument("--patience", default=6, type=int)

    parser.add_argument("--op_code", default="4", type=str)
    parser.add_argument("--slot_token", default="[SLOT]", type=str)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float)
    parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float)
    parser.add_argument("--decoder_teacher_forcing", default=0.5, type=float)
    parser.add_argument("--word_dropout", default=0.1, type=float)
    parser.add_argument("--not_shuffle_state", default=False, action='store_true')
    parser.add_argument("--shuffle_p", default=0.5, type=float)

    parser.add_argument("--n_history", default=1, type=int)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--msg", default=None, type=str)
    parser.add_argument("--exclude_domain", default=False, action='store_true')
    parser.add_argument("--train_feedback", default=1.0, type=float)

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
        args.save_dir = 'outputs/SOMDST/WOZ_outputs'
    elif args.dataset == 'MultiWOZ_2.1':
        data_root = 'data/MultiWOZ_2.1'
        config_root = 'data/dataset_config'
        feedback_data_root = 'feedback_data/MultiWOZ_2.1'
        args.train_data_path = os.path.join(data_root, 'train_dials.json')
        args.dev_data_path = os.path.join(data_root, 'dev_dials.json')
        args.test_data_path = os.path.join(data_root, 'test_dials.json')
        args.config_path = os.path.join(config_root, 'multiwoz21.json')
        args.feedback_data_path = os.path.join(feedback_data_root, 'add[4]_delete[2]_seed[42].json')
        args.save_dir = 'outputs/SOMDST/MultiWOZ_outputs'
    else:
        print('select dataset in WOZ_2.0 and MultiWOZ_2.1')
        exit()
    args.shuffle_state = False if args.not_shuffle_state else True
    print('pytorch version: ', torch.__version__)
    print(args)
    main(args)
