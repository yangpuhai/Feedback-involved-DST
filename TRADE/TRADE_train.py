import os
os.environ['CUDA_VISIBLE_DEVICES']="4"

from model.TRADE import TRADE
from torch import optim
from TRADE_evaluation import model_evaluation, model_evaluation_batch

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
    mask_sum = mask.sum()
    loss = losses.sum()
    if mask_sum != 0:
        loss = loss / mask_sum.float()
    return loss

def main(args):
    def worker_init_fn(worker_id):
        np.random.seed(args.random_seed + worker_id)
    if args.dataset == 'WOZ_2.0':
        from TRADE_utils.WOZ_data_utils import prepare_dataset, MultiWozDataset, postprocessing, state_equal, SLOT, OP
    if args.dataset == 'MultiWOZ_2.1':
        from TRADE_utils.MultiWOZ_data_utils import prepare_dataset, MultiWozDataset, postprocessing, state_equal, OP, make_slot_meta
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
    print(slot_meta)

    train_data_raw, dev_data_raw, test_data_raw, lang = prepare_dataset(
        train_data_path=args.train_data_path,
        dev_data_path=args.dev_data_path,
        test_data_path=args.test_data_path,
        feedback_data_path = args.feedback_data_path,
        train_feedback = args.train_feedback,
        slot_meta=slot_meta)
    
    train_data = MultiWozDataset(train_data_raw,
                                 lang,
                                 args.word_dropout)

    print("# train examples %d" % len(train_data_raw))
    print("# dev examples %d" % len(dev_data_raw))
    print("# test examples %d" % len(test_data_raw))

    model = TRADE(lang, args.hidden_size, args.dropout, len(op2id), slot_meta)
    model.to(device)

    # Initialize optimizers and criterion
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
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
            input_ids, op_ids, gen_ids, input_lens, max_input, max_value = batch

            if rng.random() < args.decoder_teacher_forcing:  # teacher forcing
                teacher = gen_ids
            else:
                teacher = None

            state_scores, gen_scores = model(input_ids=input_ids,
                                            input_lens=input_lens,
                                            max_value = max_value,
                                            teacher=teacher)

            loss_s = loss_fnc(state_scores.contiguous().view(-1, len(op2id)), op_ids.view(-1))
            loss_g = masked_cross_entropy_for_value(gen_scores.contiguous(),
                                                    gen_ids.contiguous(),
                                                    lang.word2index['[PAD]'])
            
            loss = loss_s + loss_g
            batch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            model.zero_grad()

            total_step += 1

            if step % 100 == 0:
                print("[%d/%d] [%d/%d] mean_loss : %.3f, state_loss : %.3f, gen_loss : %.3f" \
                        % (epoch, args.n_epochs, step,
                        len(train_dataloader), np.mean(batch_loss),
                        loss_s.item(), loss_g.item()))
                batch_loss = []

        if epoch % args.eval_epoch == 0:
            print('total_step: ',total_step)
            print('evaluate...')
            eval_res = model_evaluation_batch(args, postprocessing, state_equal, model, dev_data_raw, lang, slot_meta, OP, epoch)
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
    model = TRADE(lang, args.hidden_size, args.dropout, len(op2id), slot_meta)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt)
    model.to(device)

    model_evaluation_batch(args, postprocessing, state_equal, model, test_data_raw, lang, slot_meta, OP, best_epoch)


if __name__ == "__main__":
    print('start')
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--dataset", default='MultiWOZ_2.1', type=str)

    parser.add_argument("--random_seed", default=41, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--n_epochs", default=200, type=int)
    parser.add_argument("--eval_epoch", default=1, type=int)
    parser.add_argument("--patience", default=6, type=int)

    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--decoder_teacher_forcing", default=0.5, type=float)
    parser.add_argument("--word_dropout", default=0.1, type=float)
    parser.add_argument("--hidden_size", default=400, type=int)
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
        args.save_dir = 'outputs/TRADE/WOZ_outputs'
    elif args.dataset == 'MultiWOZ_2.1':
        data_root = 'data/MultiWOZ_2.1'
        config_root = 'data/dataset_config'
        feedback_data_root = 'feedback_data/MultiWOZ_2.1'
        args.train_data_path = os.path.join(data_root, 'train_dials.json')
        args.dev_data_path = os.path.join(data_root, 'dev_dials.json')
        args.test_data_path = os.path.join(data_root, 'test_dials.json')
        args.config_path = os.path.join(config_root, 'multiwoz21.json')
        args.feedback_data_path = os.path.join(feedback_data_root, 'add[4]_delete[2]_seed[42].json')
        args.save_dir = 'outputs/TRADE/MultiWOZ_outputs'
    else:
        print('select dataset in WOZ_2.0 and MultiWOZ_2.1')
        exit()
    print('pytorch version: ', torch.__version__)
    print(args)
    main(args)