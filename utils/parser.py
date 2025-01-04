import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='WeCromCL Stage1 Training')
    parser.add_argument("--config", type=str, default='configs/finetune/ic15.yaml', help="path to config file")
    parser.add_argument('--resume', default="ckpts/ic15_stage1_best.pth",type=str)
    return parser
    