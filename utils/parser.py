import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='WeCromCL Stage1 Training')
    parser.add_argument('--name', default='stage1', type=str)
    parser.add_argument('--exp', default='pretrain', type=str)
    parser.add_argument("--config", type=str, default='configs/pretrain/pretrain.yaml', help="path to config file")
    ## network setting
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help='resnet18, resnet34, resnet50, resnet101, resnet152')
    
    ## data setting
    parser.add_argument('--root', default='data',type=str)
    parser.add_argument('--train_input_size', default=640, type=int)
    parser.add_argument('--min_text_size', default=6, type=int)
    parser.add_argument('--pos_scale', default=0.2, type=int)
    parser.add_argument('--ignore_scale', default=0.5, type=int)
    parser.add_argument('--alphabet', default='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
    parser.add_argument('--ignore_case', default=True, type=bool)
    parser.add_argument('--max_text_length', default=25, type=int)
    parser.add_argument('--max_text_num', default=25, type=int)
    parser.add_argument('--nums_sample_point', default=25, type=int, help='nums_sample_point for rec')
    parser.add_argument('--nums_per_text', default=100, type=int, help='nums_sample_point for rec')
    ## optim setting
    parser.add_argument('--nsamples', default=64, type=int)
    parser.add_argument('--alpha_score', default=5, type=int)
    parser.add_argument('--alpha_loc', default=0, type=int)
    parser.add_argument('--alpha_mask', default=0, type=int)
    parser.add_argument('--alpha_sampled', default=0, type=int)
    parser.add_argument('--alpha_det', default=2, type=int)
    parser.add_argument('--alpha_rec', default=0, type=int)
    parser.add_argument('--alpha_bbox', default=0, type=int)
    parser.add_argument('--cls_loss', default='dice', type=str)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--focal_gamma', default=0.1, type=float)
    parser.add_argument('--loc_loss', default='iou', type=str)
    parser.add_argument('--focal_iou_gamma', default=5, type=float)
    parser.add_argument('--fpn_strides', default=[4, 8], type=float)
    parser.add_argument('--use_stride', default=8, type=int)
    parser.add_argument('--false_label_num', default=16, type=int)
    ## optim setting
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--resume',
                        default=None,type=str)
    parser.add_argument('--start_iter', default=0, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)  ## bs 32 1e-2  bs 16 5e-3
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--optim', default='sgd', type=str, help='sgd, adam, adadelta')
    parser.add_argument('--lr_policy', default='poly', type=str, help='step, poly')
    parser.add_argument('--niter', default=170000, type=int)
    parser.add_argument('--clip_grad', default=False, type=bool)
    ## output setting
    parser.add_argument('--log_freq', default=10, type=int)
    parser.add_argument('--save_folder', default='weights/', type=str)
    parser.add_argument('--spotting', default='mean', type=str)
    ## extra setting
    parser.add_argument('--use_bbox_loss', default=False, type=bool)
    parser.add_argument('--bbox_for_boundary_points', default=False, type=bool)
    parser.add_argument('--transform_method', default='minmax', type=str)
    parser.add_argument('--is_training', default=True, type=bool)
    parser.add_argument('--is_finetune', default=True, type=bool)
    parser.add_argument('--joint_finetune', default=False, type=bool)
    parser.add_argument('--joint_pretrain', default=False, type=bool)
    parser.add_argument('--deformable_layer', default=6, type=int)
    parser.add_argument('--num_layer', default=4, type=int)
    parser.add_argument('--accumulate', default=2, type=int)
    parser.add_argument('--multi_scale', default=True, type=bool)
    parser.add_argument('--with_deformable_encoder', default=True, type=int)
    
    parser.add_argument('--embed_dim', default=512, type=int)
    parser.add_argument('--vision_heads', default=1, type=int)
    parser.add_argument('--image_resolution', default=512, type=int)
    parser.add_argument('--vision_width', default=64, type=int)
    parser.add_argument('--transformer_width', default=512, type=int)
    parser.add_argument('--transformer_layers', default=12, type=int)
    parser.add_argument('--transformer_heads', default=8, type=int)
    parser.add_argument('--vocab_size', default=70, type=int)
    parser.add_argument('--maintain', default=4, type=int)
    return parser
    