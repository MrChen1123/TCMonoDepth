import argparse


def parser_paras():
    # Settings
    parser = argparse.ArgumentParser(description="This is a PyTorch Implementation of TCMonoDepth.")
    # model params
    parser.add_argument('--model', default='large', choices=['small', 'large', 'dpt-large', 'dpt-hybrid'], help='size of the model')
    parser.add_argument('--backbone', default='vitl16_384', choices=['vitl16_384', 'vitb_rn50_384', 'vitb16_384'], help='size of the model')
    parser.add_argument('--resume', type=str, default='./weights/_ckpt_small.pt.tar', help='path to checkpoint file')

    # loss
    parser.add_argument('--alpha', type=float, default=0.2, help='weighted to tc_loss')
    parser.add_argument('--lam', type=float, default=1.0, help='weighted to tc_loss')
    parser.add_argument('--bata', type=float, default=1.0, help='weighted to tc_loss')
    parser.add_argument('--weight_reg', type=float, default=0.5, help='weighted to tc_loss')
    parser.add_argument('--variance_focus',  type=float, default=0.85, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error')

    # train params
    parser.add_argument('--lr', type=float, default=1e-5, help='weighted to tc_loss')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weighted to tc_loss')
    parser.add_argument('--epoch', type=float, default=20, help='weighted to tc_loss')
    parser.add_argument('--log_freq', type=float, default=10, help='weighted to tc_loss')
    parser.add_argument('--depth_model_path', type=str, default="./weights/midas_v21-f6b98070.pt", help='weighted to tc_loss')
    parser.add_argument('--opt_path', type=str, default="", help='weighted to tc_loss')

    # Data
    parser.add_argument('--data_root', default='', type=str, help='video root path')
    parser.add_argument('--train_file', default="./train_out_small_1_10.json", type=str, help='video train file')
    parser.add_argument('--val_file', default="./test_out_small.json", type=str, help='video val file')
    parser.add_argument('--batch_size', default=160, type=int, help='video val file')
    parser.add_argument('--num_workers', default=10, type=int, help='video val file')

    parser.add_argument('--input', default='./videos1', type=str, help='video root path')
    parser.add_argument('--output', default='./output2', type=str, help='path to save output')
    parser.add_argument('--resize', type=int, default=[448, 320],
                        help="spatial dimension to resize input (default: small model:256, large model:384)")
    parser.add_argument('--input_size', type=int, default=[320, 224],
                        help="spatial dimension to resize input (default: small model:256, large model:384)")

    parser.add_argument('--degree', default=5, type=float)
    parser.add_argument('--port', default='23457', type=str)

    # RAFT_model
    parser.add_argument('--raft_model_path', default='./weights/raft-things.pth', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    # save_dir
    parser.add_argument('--save_dir', default='./checkpoints', help='use efficent correlation implementation')
    args = parser.parse_args()
    return args