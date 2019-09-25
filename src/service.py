import sys
from data import Data
from segmentation import BeamSeg
from segeval import window_diff

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

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("ERROR: provide <in_data_path> <out_path> arguments")
        
    else:
        in_data_path = sys.argv[1]
        out_path = sys.argv[2]
        d = Data(in_data_path, max_features=100, lemmatize=True) #TODO: add argument for config
        model = BeamSeg(d, "cfg.json") #TODO: add argument for config
        model.segment_docs()
        hyp_seg = model.get_final_segmentation(0)
        ref_seg = d.rho
        print("WD %f"%(wd(hyp_seg, ref_seg)))
        