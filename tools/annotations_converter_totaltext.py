import os
import json

def converter_from_txt_to_json(txt_dir,json_file):
    json_data = {}
    for txt_file in os.listdir(txt_dir):
        with open(os.path.join(txt_dir, txt_file), 'r') as f:
            lines = f.readlines()
        
        img_name = (txt_file[:-4]+".jpg").zfill(11)
        words = []
        for line in lines:
            word = line.strip().split(",####")[-1]
            if word != "###":
                words.append(word)
        if len(words) != 0:
            json_data[img_name] = words
    with open(json_file, 'w') as f:
        json.dump(json_data, f)
if __name__ == "__main__":
    txt_dir = "/data/wujingjing/data_copy/totaltext/gts_test_abc"
    json_file = "dataset/annotations/totaltext_test.json"
    converter_from_txt_to_json(txt_dir,json_file)
    