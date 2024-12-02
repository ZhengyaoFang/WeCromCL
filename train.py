import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import time
import math
import torch
import logging
from omegaconf import OmegaConf
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from utils.logger import setup_logger, print_args
from utils.parser import get_parser
from utils.str_label_converter import StrLabelConverter
from utils.trainer import Trainer
from models.loss import ContrastiveLoss
from models.wecromcl import WeCromCL
import torch.utils.data as data

from data.dataset import TrainDataset, TextCollate

def make_log_dir(args):
    if os.path.exists(args.save_folder) == False:
        os.mkdir(args.save_folder)
    args.save_folder = args.save_folder + args.name + '/'
    if os.path.exists(args.save_folder) == False:
        os.mkdir(args.save_folder)
    args.save_folder = args.save_folder + args.exp + '/'
    if os.path.exists(args.save_folder) == False:
        os.mkdir(args.save_folder)

    log_file_path = args.save_folder + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log'
    setup_logger(log_file_path)
    print_args(args)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    cfgs = OmegaConf.load(args.config)
    base_cfg = OmegaConf.load(cfgs._base_)
    cfgs = OmegaConf.merge(base_cfg, cfgs)
    cfgs.trainer.save_folder = args.save_folder

    make_log_dir(args)

    criterion = ContrastiveLoss()
    converter = StrLabelConverter(cfgs.data.alphabet, cfgs.data.ignore_case, cfgs.max_text_length)
    net = WeCromCL(cfgs)
    
    net = torch.nn.DataParallel(net).cuda()
    if args.resume:
        logging.info('Resuming training, loading {}...'.format(args.resume))
        net.load_state_dict(torch.load(args.resume))

    ## setup optimizer
    if cfgs.trainer.optim == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay)
        logging.info('models will be optimed by sgd')
    elif cfgs.trainer.optim == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        logging.info('models will be optimed by adam')
    elif cfgs.trainer.optim == 'adadelta':
        optimizer = optim.Adadelta(net.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        logging.info('models will be optimed by adam')

    if cfgs.trainer.lr_policy == 'poly':
        logging.info("Model will be train from iter {}".format(cfgs.trainer.start_iter))
        def lambda_rule(epoch):
            lr_l = math.pow(1 - (epoch + cfgs.trainer.start_iter)* 1.0 / cfgs.trainer.niter, 0.9)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif cfgs.trainer.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=cfgs.trainer.lr_decay_iters, gamma=0.1)
    elif cfgs.trainer.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif cfgs.trainer.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfgs.trainer.niter, eta_min=0)
    else:
        exit('learning rate policy [%s] is not implemented', cfgs.trainer.lr_policy)

    cudnn.benchmark = True
    net.train()
    dataset = TrainDataset(cfgs)
    dataloader = data.DataLoader(dataset, batch_size=cfgs.data.train.batch_size, num_workers=cfgs.data.train.num_workers, shuffle=True, collate_fn=TextCollate(cfgs, dataset), pin_memory=True)
    trainer = Trainer(net, optimizer, scheduler, criterion, dataloader, converter, cfgs)
    trainer.train()

    

