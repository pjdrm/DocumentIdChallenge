import sys
from data import Data
import segmentation as seg
from segeval import window_diff
import json

def segeval_converter(segmentation):
    segeval_format = []
    sent_count = 0
    for sent in segmentation:
        if sent == 1:
            segeval_format.append(sent_count+1)
            sent_count = 0
        else:
            sent_count += 1
    segeval_format.append(sent_count)
    return segeval_format

def wd(hyp_seg, ref_seg):
    hyp_seg = segeval_converter(hyp_seg)
    ref_seg = segeval_converter(ref_seg)
    wd = window_diff(hyp_seg, ref_seg)
    return float(wd)

def split_data(paragraphs, n):
    for i in range(0, len(paragraphs), n):  
        yield paragraphs[i:i + n]

def get_gs_seg(paragraphs):
    ref_seg = [0]
    for i, para in enumerate(paragraphs[1:]):
        if para["document_start"]:
            ref_seg[i] = 1
        ref_seg.append(0)
    ref_seg[-1] = 1
    return ref_seg

if __name__ == "__main__":
    '''
    Service to perform document identification on an input file.
    '''

    if len(sys.argv) != 3:
        print("ERROR: provide <in_data_path> <out_path> arguments")
        
    else:
        in_data_path = sys.argv[1]
        out_path = sys.argv[2]
        with open("configs.json") as f:
            seg_config = json.load(f)#TODO: add argument for config
        
        with open(in_data_path) as data_file:    
            paragraphs = json.load(data_file)
        
        n = 70
        max_features = seg_config["max_features"]
        data_chunks = list(split_data(paragraphs, n))
        seg.TOTAL_CHUNKS = len(data_chunks)
        hyp_seg = []
        for i, paragraph_chunk in enumerate(data_chunks):
            d = Data(paragraph_chunk, max_features=max_features, lemmatize=True)
            model = seg.BeamSeg(d, seg_config)
            model.segment_docs()
            chunk_seg_hyp = model.get_final_segmentation(0)
            chunk_seg_hyp[-1] = 0
            hyp_seg += chunk_seg_hyp
            seg.CHUNK_I += 1
        hyp_seg[-1] = 1
        
        if "document_start" in paragraphs[0]: #Its development data
            ref_seg = get_gs_seg(paragraphs)
            #print("ref: %s\nhyp: %s"%(ref_seg, hyp_seg))
            print("WD %f"%(wd(hyp_seg, ref_seg)))
        
        pred_out = []
        flag = False
        for rho in hyp_seg:
            if flag:
                flag = False
                continue
            if rho == 1:
                pred_out.append("false")
                pred_out.append("true")
                flag = True
            else:
                pred_out.append("false")
        pred_out[0] = "true" #First one is always true
        pred_out.pop()
        with open(out_path, "w+") as f:
            f.write("["+", ".join(pred_out)+"]")
