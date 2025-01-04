import os
import cv2
import json
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
import torch.utils.data as data
from models.wecromcl import WeCromCL
from utils.parser import get_parser
from utils.str_label_converter import StrLabelConverter
from data.dataset import TestDataset


def _nms(heat, kernel=5):
    pad = (kernel - 1) // 2

    hmax = torch.nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def main(cfgs, args):
    net = WeCromCL(cfgs)
    if args.resume:
        net = torch.nn.DataParallel(net).cuda()
        print("Loading {}...".format(args.resume))
        net.load_state_dict(torch.load(args.resume))
        net = net.module
    net.eval()
    converter = StrLabelConverter(cfgs.data.alphabet, cfgs.data.ignore_case, cfgs.max_text_length)

    dataset = TestDataset(cfgs)
    results = {}

    with torch.no_grad():
        for i, examples in enumerate(tqdm(dataset)):
            raw_texts = examples['recs']
            test_h, test_w = examples['image'].shape[1:]
            ori_w, ori_h = examples['image_size']
            ratio_h = float(test_h) / ori_h
            ratio_w = float(test_w) / ori_w

            map_texts = {}
            for j in range(len(raw_texts)):
                text_key = raw_texts[j].lower()
                if text_key not in map_texts.keys():
                    map_texts[text_key] = {
                        'num': 1, 
                        'raw_texts': [raw_texts[j],], 
                        'points': [],
                        'logits': []
                        }
                else:
                    map_texts[text_key]['num'] += 1
                    map_texts[text_key]['raw_texts'].append(raw_texts[j])
            
            text_embeds = torch.tensor([converter.encode(text)[0] for text in map_texts.keys()]).long().cuda()

            img = examples['image'].unsqueeze(0).cuda()
            img_feature = net.channel_to_512(net.encode_image(img))
            H, W = img_feature.size(2), img_feature.size(3)
            text_features = net.encode_text(text_embeds)
            attn_maps = net.attn_pool(text_features.unsqueeze(1), img_feature)[1].view(1,-1,H,W)
            attn_maps = _nms(attn_maps)[0]
            for j, text_key in enumerate(map_texts.keys()):
                attn_map = attn_maps[j]
                text_num = map_texts[text_key]['num']
                topk_logits, topk_proposal = torch.topk(attn_map.view(1, -1), text_num)

                for p in range(topk_proposal.size(1)):
                    y = int(topk_proposal[0][p] / W)
                    x = int(topk_proposal[0][p] - y * W)

                    x = (x * cfgs.model.use_stride + cfgs.model.use_stride // 2) / ratio_w
                    y = (y * cfgs.model.use_stride + cfgs.model.use_stride // 2) / ratio_h
                    
                    map_texts[text_key]['points'].append((x, y))
                    map_texts[text_key]['logits'].append(topk_logits[0][p].item())
            
            final_texts = []
            final_points = []
            final_scores = []
            for j, text_key in enumerate(map_texts.keys()):
                for k in range(map_texts[text_key]['num']):
                    final_texts.append(map_texts[text_key]['raw_texts'][k])
                    final_points.append(map_texts[text_key]['points'][k])
                    final_scores.append(map_texts[text_key]['logits'][k])
            
            img_path = examples['image_path']
            img_name = img_path.replace(cfgs.data.test.img_rootdir, "", 1)
            img_name = img_name[1:] if img_name[0] == '/' else img_name
            results[img_name] = {
                'text': final_texts,
                'points': final_points,
                'scores': final_scores
            }
    
    print('Save results to {}'.format(cfgs.inference.output_json_path))
    output_path = cfgs.inference.output_json_path
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(cfgs.inference.output_json_path, 'w') as f:
        json.dump(results, f)

    if cfgs.inference.vis:
        print("Visualizing results to {}".format(cfgs.inference.output_vis_dir))
        if not os.path.exists(cfgs.inference.output_vis_dir):
            os.makedirs(cfgs.inference.output_vis_dir)
        for img_name in tqdm(results.keys()):
            img_path = os.path.join(cfgs.data.test.img_rootdir, img_name)
            img = cv2.imread(img_path)
            for i in range(len(results[img_name]['text'])):
                text = results[img_name]['text'][i]
                point = results[img_name]['points'][i]
                cv2.circle(img, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
                cv2.putText(img, text, (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.imwrite(os.path.join(cfgs.inference.output_vis_dir, os.path.basename(img_name)), img)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    cfgs = OmegaConf.load(args.config)
    base_cfg = OmegaConf.load(cfgs._base_)
    cfgs = OmegaConf.merge(base_cfg, cfgs)

    main(cfgs, args)
    

                
                    

