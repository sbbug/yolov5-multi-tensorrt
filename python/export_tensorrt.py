import tensorrt as trt
import sys
import argparse

"""
takes in onnx model
converts to tensorrt
"""

def cli():
    desc = 'compile Onnx model to TensorRT'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-m', '--model', default="yolov5l-visible.onnx",help='onnx file location inside ./lib/models')
    parser.add_argument('-fp', '--floatingpoint', type=int, default=32, help='floating point precision. 16 or 32')
    parser.add_argument('-o', '--output', help='name of trt output file')
    args = parser.parse_args()
    model = args.model
    fp = args.floatingpoint
    if fp != 16 and fp != 32:
        print('floating point precision must be 16 or 32')
        sys.exit()
    output = args.output or 'yolov5l-visible-simple-{}.trt'.format(fp)
    return {
        'model': model,
        'fp': fp,
        'output': output
    }

if __name__ == '__main__':
    args = cli()
    model = '/home/user/Code/yolov5-tensorrt-master/python/lib/models/{}'.format(args['model'])
    output = 'lib/models/{}'.format(args['output'])
    logger = trt.Logger(trt.Logger.VERBOSE)
    EXPLICIT_BATCH = []
    print('trt version', trt.__version__)
    if trt.__version__[0] >= '7':
        EXPLICIT_BATCH.append(
            1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    with trt.Builder(logger) as builder, builder.create_network(*EXPLICIT_BATCH) as network, trt.OnnxParser(network, logger) as parser:
        builder.max_workspace_size = 1 << 32
        builder.max_batch_size = 1
        if args['fp'] == '16':
            builder.fp16_mode = True

        with open(model, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
            
        # reshape input from 32 to 1
        shape = list(network.get_input(0).shape)
        engine = builder.build_cuda_engine(network)
        with open(output, 'wb') as f:
            f.write(engine.serialize())
