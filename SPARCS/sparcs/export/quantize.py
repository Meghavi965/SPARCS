import argparse
from onnxruntime.quantization import quantize_dynamic, QuantType

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('input'); ap.add_argument('output'); args=ap.parse_args(); quantize_dynamic(args.input,args.output,weight_type=QuantType.QInt8,per_channel=True,reduce_range=False)
if __name__=='__main__': main()
