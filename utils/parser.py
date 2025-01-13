import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='WeCromCL Stage1 Training')
    parser.add_argument("--config", type=str, default=None, help="path to config file")
    parser.add_argument('--resume', default=None,type=str)
    return parser
    