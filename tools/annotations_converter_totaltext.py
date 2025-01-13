import os
import json
import numpy as np
from shapely.geometry import LineString

def converter_from_txt_to_json_train(txt_dir,json_file):
    json_data = {}
    for txt_file in os.listdir(txt_dir):
        with open(os.path.join(txt_dir, txt_file), 'r') as f:
            lines = f.readlines()
        
        img_name = txt_file[:-4]+".jpg"
        words = []
        for line in lines:
            word = ",".join(line.strip().split(",")[32:])
            if word != "###" and len(word) >= 3:
                words.append(word)
        if len(words) != 0:
            json_data[img_name] = words
    with open(json_file, 'w') as f:
        json.dump(json_data, f)

def converter_from_txt_to_json_test(txt_dir, json_file):
    json_data = {}
    for txt_file in os.listdir(txt_dir):
        with open(os.path.join(txt_dir, txt_file), 'r') as f:
            lines = f.readlines()
        
        img_name = txt_file[:-4].zfill(7)+".jpg"
        words = []
        for line in lines:
            word = line.strip().split(",####")[-1]
            if word != "###" and len(word) >= 3:
                words.append(word)
        if len(words) != 0:
            json_data[img_name] = words
    with open(json_file, 'w') as f:
        json.dump(json_data, f)

def converter_to_eval_gts_train(txt_dir,json_file):
    def poly_center(poly_pts):
        poly_pts = np.array(poly_pts).reshape(-1, 2)
        num_points = poly_pts.shape[0]
        line1 = LineString(poly_pts[int(num_points / 2):])
        line2 = LineString(poly_pts[:int(num_points / 2)])
        mid_pt1 = np.array(line1.interpolate(0.5, normalized=True).coords[0])
        mid_pt2 = np.array(line2.interpolate(0.5, normalized=True).coords[0])
        return (mid_pt1 + mid_pt2) / 2
    json_data = {}
    for txt_file in os.listdir(txt_dir):
        with open(os.path.join(txt_dir, txt_file), 'r') as f:
            lines = f.readlines()
        
        img_name = txt_file[:-4]+".jpg"
        final_texts = []
        final_points = []

        for line in lines:
            word = ",".join(line.strip().split(",")[32:])
            if word != "###" and len(word) >= 3:
                final_texts.append(word)
                coords = line.strip().split(",")[:32]
                coords = [c.strip("\ufeff").strip("\xef\xbb\xbf") for c in coords]
                coords = [int(ele) for ele in coords]
                center_pt = poly_center(coords)
                final_points.append((int(center_pt[0]), int(center_pt[1])))

        if len(final_texts) != 0:
            json_data[img_name] = {
                    'text': final_texts,
                    'points': final_points,
                }
            
    with open(json_file, 'w') as f:
        json.dump(json_data, f)

def converter_to_eval_gts_test(txt_dir,json_file):
    def poly_center(poly_pts):
        poly_pts = np.array(poly_pts).reshape(-1, 2)
        num_points = poly_pts.shape[0]
        line1 = LineString(poly_pts[int(num_points / 2):])
        line2 = LineString(poly_pts[:int(num_points / 2)])
        mid_pt1 = np.array(line1.interpolate(0.5, normalized=True).coords[0])
        mid_pt2 = np.array(line2.interpolate(0.5, normalized=True).coords[0])
        return (mid_pt1 + mid_pt2) / 2
    json_data = {}
    for txt_file in os.listdir(txt_dir):
        with open(os.path.join(txt_dir, txt_file), 'r') as f:
            lines = f.readlines()
        
        img_name = txt_file[:-4].zfill(7)+".jpg"
        final_texts = []
        final_points = []

        for line in lines:
            word = line.strip().split(",####")[-1]
            if word != "###" and len(word) >= 3:
                final_texts.append(word)
                coords = line.strip().split(",")[:2]
                coords = [c.strip("\ufeff").strip("\xef\xbb\xbf") for c in coords]
                coords = [int(ele) for ele in coords]
                center_pt = coords
                final_points.append((int(center_pt[0]), int(center_pt[1])))

        if len(final_texts) != 0:
            json_data[img_name] = {
                    'text': final_texts,
                    'points': final_points,
                }
            
    with open(json_file, 'w') as f:
        json.dump(json_data, f)

if __name__ == "__main__":
    train_txt_dir = "/data/wujingjing/data_copy/totaltext/gts_abc"
    train_json_file = "/data/fzy/WeCromCL/Stage1/dataset/annotations/totaltext_train.json"
    converter_from_txt_to_json_train(train_txt_dir, train_json_file)
    test_txt_dir = "/data/wujingjing/data_copy/totaltext/gts_test_abc"
    test_json_file = "//data/fzy/WeCromCL/Stage1/dataset/annotations/totaltext_test.json"
    converter_from_txt_to_json_test(test_txt_dir, test_json_file)
    train_gts_file = "/data/fzy/WeCromCL/Stage1/dataset/eval_gts/totaltext_train_gts.json"
    converter_to_eval_gts_train(train_txt_dir, train_gts_file)
    test_gts_file = "/data/fzy/WeCromCL/Stage1/dataset/eval_gts/totaltext_test_gts.json"
    converter_to_eval_gts_test(test_txt_dir, test_gts_file)
