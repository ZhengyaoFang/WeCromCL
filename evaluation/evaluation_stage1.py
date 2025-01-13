import copy
import json
import argparse

from tqdm import tqdm
from shapely.geometry import Point

def read_gt(gt_file):
    with open(gt_file, 'r') as f:
        gts = json.load(f)

    gt_dict = {}
    for gt in gts:
        points = [Point(pt[0], pt[1]) for pt in gts[gt]['points']]
        recs = gts[gt]['text']
        matched = [0] * len(recs)
        gt_dict[gt] = [points, recs, matched]
    return gt_dict

def read_result(result_file):
    with open(result_file, 'r') as f:
        results = json.load(f)
    new_results = {
        'img_name':[],
        'recs':[],
        'points':[],
        'scores':[],
    }
    for img_name in results:
        for i in range(len(results[img_name]['text'])):
            new_results['img_name'].append(img_name)
            new_results['recs'].append(results[img_name]['text'][i])
            new_results['points'].append(results[img_name]['points'][i])
            new_results['scores'].append(results[img_name]['scores'][i])

    return new_results

def evaluate(results, gts, conf_thres, pbar):
    gts = copy.deepcopy(gts)
    results = copy.deepcopy(results)

    ngt = sum([len(ele[0]) for ele in gts.values()])
    ndet = 0
    ntp = 0

    for i in range(len(results['img_name'])):
        confidence = results['scores'][i]
        if confidence < conf_thres:
            continue
        img_name = results['img_name'][i]
        pred_coords = results['points'][i]
        pred_rec = results['recs'][i]
        pred_point = Point(pred_coords[0], pred_coords[1])

        gt_points = gts[img_name][0]
        gt_recs = gts[img_name][1]
        gt_matched = gts[img_name][2]

        dists = [pred_point.distance(gt_point) for gt_point in gt_points]
        minvalue = min(dists)
        idmin = dists.index(minvalue)
        if pred_rec.lower() == gt_recs[idmin].lower() and gt_matched[idmin] == 0:
            gt_matched[idmin] = 1
            ntp += 1
        ndet += 1
        pbar.update(1)


    if ndet == 0 or ntp == 0:
        recall = 0
        precision = 0
        hmean = 0
    else:
        recall = ntp / ngt
        precision = ntp / ndet
        hmean = 2 * recall * precision / (recall + precision)
    
    return precision, recall, hmean, ntp, ngt, ndet

def main(args):
    print('Reading GT: {}'.format(args.eval_gts))
    print('Reading Result: {}'.format(args.result))
    gts = read_gt(args.eval_gts)
    results = read_result(args.result)
    conf_thresh_list = [0,0.01,0.02,0.03,0.04,0.05]

    hmeans = []
    recalls = []
    precisions = []
    print('Evaluating...')
    with tqdm(total=len(conf_thresh_list) * len(results)) as pbar:
        for k in conf_thresh_list:
            precision, recall, hmean, pgt, ngt, ndet = evaluate(
                results=results,
                gts=gts,
                conf_thres = k,
                pbar = pbar,
            )
            hmeans.append(hmean)
            recalls.append(recall)
            precisions.append(precision)

    
    max_hmean = max(hmeans)
    max_hmean_index = len(hmeans) - hmeans[::-1].index(max_hmean) - 1
    precision = precisions[max_hmean_index]
    recall = recalls[max_hmean_index]
    conf_thres = conf_thresh_list[max_hmean_index]
    print(f'MAX:Precision: {precision:.4f}, Recall: {recall:.4f}, Hmean: {max_hmean:.4f}, Conf Thres: {conf_thres:.4f}')

def get_parser():
    parser = argparse.ArgumentParser(description="Evaluation of the model")
    parser.add_argument("--result", type=str, default="inference/totaltext_train.json", help="Path to the inference result file end with '.json'.")
    parser.add_argument("--eval_gts", type=str, default="dataset/eval_gts/totaltext_train_gts.json", help="Path to the ground truth file end with '.json'.")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_parser()
    main(args)