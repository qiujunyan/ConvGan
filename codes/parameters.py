from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--epoch_num", type=int, default=1000)
parser.add_argument("--embed_dim", type=int, default=64)
parser.add_argument("--hidden_dim", type=int, default=64)
parser.add_argument("--ans_max_len", type=int, default=40)
parser.add_argument("--dia_max_len", type=int, default=400)
parser.add_argument("--channels", type=list, default=[16, 32])
parser.add_argument("--g_lr", type=float, default=1e-2)
parser.add_argument("--d_lr", type=float, default=1e-2)
parser.add_argument("--g_dropout", type=float, default=0.3)
parser.add_argument("--clip", type=float, default=1)
parser.add_argument("--decay_rate", type=float, default=0.2)
parser.add_argument("--curriculum_rate", type=float, default=0)
parser.add_argument("--reg_rate", type=float, default=1e-3,
                    help="regularization rate")
parser.add_argument("--train_dir1", type=str, default="./data/mutual/train/1",
                    help="GAN model training data")
parser.add_argument("--train_dir2", type=str, default="./data/mutual/train/2",
                    help="Generator pre-training")
parser.add_argument("--test_dir", type=str, default="./data/mutual/test")
parser.add_argument("--dev_dir", type=str, default="./data/mutual/dev")
parser.add_argument("--model_dir", type=str, default="./model")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--k", type=int, default=5,
                    help="train discriminator after training generator k times")
parser.add_argument("--n_times", type=int, default=10,
                    help="rollout algorithm parameter")
parser.add_argument("--num_samples", type=int, default=5)


args = parser.parse_args()
