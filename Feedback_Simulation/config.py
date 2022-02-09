import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="WOZ_2.0", help="MultiWOZ_2.2 or WOZ_2.0")
    parser.add_argument("--model_checkpoint", type=str, default="t5-small", help="Path, url or short name of the model")
    parser.add_argument("--saving_dir", type=str, default="save", help="Path for saving")
    parser.add_argument("--train_batch_size", type=int, default=10, help="Batch size for training")
    parser.add_argument("--dev_batch_size", type=int, default=10, help="Batch size for validation")
    parser.add_argument("--test_batch_size", type=int, default=10, help="Batch size for test")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--length", type=int, default=50, help="max length for decoding")
    parser.add_argument("--GPU", type=int, default=3, help="number of gpu to use")
    parser.add_argument("--mode", type=str, default="train")

    args = parser.parse_args()
    return args
