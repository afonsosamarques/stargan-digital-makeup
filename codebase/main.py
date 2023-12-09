import argparse
import os

import torch

from data_utils.data_loader import get_loader
from solver import Solver


def run(config):
    # For fast training
    torch.backends.cudnn.benchmark = True

    # Create directories if not exist
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)

    if not os.path.exists(config.debug_dir):
        os.makedirs(config.debug_dir)

    if not os.path.exists(config.test_results_dir):
        os.makedirs(config.test_results_dir)

    # Setup data loaders
    nomakeup_loader = get_loader(
        config.nomakeup_dir, 
        config.image_crop_size, 
        config.image_final_size,
        'NoMakeup', 
        config.nomakeup_file_path, 
        batch_size=config.batch_size,
        mode=config.mode, 
        test_set=config.nomakeup_test_set,
        num_workers=config.num_workers
    )

    makeup_loader = get_loader(
        config.makeup_dir, 
        config.image_crop_size, 
        config.image_final_size,
        'Makeup', 
        config.makeup_att_path, 
        batch_size=config.batch_size,
        mode=config.mode, 
        num_workers=config.num_workers
    )

    # Setup solver for training or testing
    solver = Solver(nomakeup_loader, makeup_loader, config)

    if config.mode == 'train':
        solver.train()

    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #
    # High-Level configuration
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--use_tensorboard', type=str, default='true')

    #
    # Model configuration
    parser.add_argument('--generator_type', type=str, default='drnet', help='generator architecture to be used: DRUNET or DRNET')
    parser.add_argument('--block_type', type=str, default='StandardBlock', help='type of block in Generator network: StandardBlock or DeepBlock')
    parser.add_argument('--makeup_label_size', type=int, default=4, help='number of makeup products in the makeup dataset')
    parser.add_argument('--att_depth', type=int, nargs='+', default=[4,4,2], help='number of possible categories for non-binary attributes')
    parser.add_argument('--image_crop_size', type=int, default=128, help='center crop size for input images')
    parser.add_argument('--image_final_size', type=int, default=128, help='input image resolution')
    parser.add_argument('--g_init_conv', type=int, default=64, help='number of channels in first conv layer of G')
    parser.add_argument('--d_init_conv', type=int, default=64, help='number of channels in first conv layer of D')
    parser.add_argument('--g_std_blocks', type=int, default=0, help='number of standard blocks in G')
    parser.add_argument('--downsampling_factor', type=int, default=4, help='input downsampling factor if DRUNET')
    parser.add_argument('--g_res_blocks', type=int, default=4, help='number of residual blocks in G (for one tower if DRUNET)')
    parser.add_argument('--d_depth', type=int, default=5, help='depth of D')
    parser.add_argument('--lambda_adv', type=float, default=1, help='weight for adversarial loss')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty') 

    #
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=8, help='mini-batch size per dataset')
    parser.add_argument('--num_iters', type=int, default=340800, help='number of iterations for training D')
    parser.add_argument('--g_train_step', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--makeup_removal_step', type=int, default=2, help='number of makeup to nomakeup iterations per nomakeup to makeup iterations')
    parser.add_argument('--iters_bef_decay', type=int, default=170400, help='number of iterations before starting learning rate decay')
    parser.add_argument('--g_lr', type=float, default=0.00005, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--lr_update_step', type=int, default=284, help='frequency of learning rate decay')
    parser.add_argument('--g_decay_rate', type=float, default=0.001667, help='learning rate decay for G (%)')
    parser.add_argument('--d_decay_rate', type=float, default=0.001667, help='learning rate decay for D (%)')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimiser')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimiser')
    parser.add_argument('--resume_iter', type=int, default=0, help='resume training from this step')
    parser.add_argument('--lambda_recalib_step', type=int, default=119280, help='step for increasing adversarial and domain classification loss weights')
    parser.add_argument('--lambda_recalib_factor', type=int, default=5, help='recalibrate weight of adversarial and domain classification losses vs reconstruction loss')
    parser.add_argument('--log_step', type=int, default=1420, help='frequency for logging training results')
    parser.add_argument('--debug_step', type=int, default=28400, help='frequency for debugging during training')
    parser.add_argument('--model_ckpt_step', type=int, default=28400, help='frequency for storing model checkpoints')

    #
    # Testing configuration
    parser.add_argument('--nomakeup_test_set', type=int, default=56, help='size of the test set')

    #
    # Directory configuration
    parser.add_argument('--nomakeup_dir', type=str, help='directory for NoMakeup images')
    parser.add_argument('--makeup_dir', type=str, help='directory for Makeup images')
    parser.add_argument('--makeup_att_path', type=str, help='file for Makeup labels')
    parser.add_argument('--nomakeup_file_path', type=str, help='file for NoMakeup txt file')
    parser.add_argument('--log_dir', type=str, help='directory for storing training logs')
    parser.add_argument('--model_dir', type=str, help='directory for storing model checkpoints')
    parser.add_argument('--debug_dir', type=str, help='directory for storing debugging results')
    parser.add_argument('--test_results_dir', type=str, help='directory for storing test results')

    #
    # Run
    config = parser.parse_args()
    print(f"Config:\n{config}")
    run(config)
