from scipy import io as sio
import json
from tqdm import tqdm

def converter_from_mat_to_json(mat_file, json_file):
    targets = {}
    sio.loadmat(mat_file, targets, squeeze_me=True, struct_as_record=False,variable_names=['imnames', 'wordBB', 'txt', 'charBB'])
    img_names = targets["imnames"]
    transcripts = targets["txt"]

    if isinstance(transcripts, str):
        transcripts = [transcripts]
    
    json_data = {}
    for i in tqdm(range(len(img_names))):
        new_transcripts = []
        for line in transcripts[i]:
            for words in line.split("\n"):
                new_transcripts += words.strip().split()
        json_data[img_names[i]] = new_transcripts
    with open(json_file, 'w') as f:
        json.dump(json_data, f)

if __name__ == "__main__":
    mat_file = "/data/wujingjing/data_copy/SynthText/gt.mat"
    json_file = "/data/fzy/WeCromCL/Stage1/data/annotations/synth800k.json"
    converter_from_mat_to_json(mat_file, json_file)
    