import argparse

def config():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--load_model",
                        action="store_true",
                        help="whether to load the pretrained model")
    parser.add_argument("--pretrained_model",
                        type=str,
                        help="whether to load the pretrained model")
    parser.add_argument("--model_name",
                        type=str,
                        required=True)
    parser.add_argument("--mode",
                        type=str,
                        required=True,
                        help="whether to train or test")
    # train
    parser.add_argument("--train_batch_size",
                        default=None,
                        type=int,
                        required=True)
    parser.add_argument("--val_batch_size",
                        default=None,
                        type=int,
                        required=True)
    parser.add_argument("--test_batch_size",
                        default=None,
                        type=int,
                        required=True)
    parser.add_argument("--eval_steps",
                        default=None,
                        type=int,
                        required=True)
    parser.add_argument("--num_train_epochs",
                        type=int,
                        required=True,
                        help="How many epochs in training process")
    parser.add_argument("--threshold",
                        type=float,
                        required=True,
                        help="below the threshold, the pixel is regarded as background. Otherwise crack")
    parser.add_argument("--pos_weights",
                        type=int,
                        required=True,
                        help="the proportion between positive and negative samples")
    parser.add_argument("--lr",
                        type=float,
                        required=True,
                        help="the learning rate of optimizer")
    parser.add_argument("--loss_weight",
                        type=float,
                        required=True,
                        help="the weight between two kinds of loss")
    parser.add_argument("--deep_supervision_weight",
                        type=float,
                        required=True,
                        help="the weight between main/sub loss")
    # environment
    parser.add_argument("--num_workers",
                        default=None,
                        type=int,
                        required=True)
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1)
    args = parser.parse_args()
    return args