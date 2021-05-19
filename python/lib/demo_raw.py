import cv2
import sys
import argparse

from ProcessorMy import Processor
from Visualizer import Visualizer

batch_img = "/home/user/Code/uva/multi_detection/data/images/visible"


def cli():
    desc = 'Run TensorRT yolov5 visualizer'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-m', '--model', default='yolov5s-simple-32.trt', help='trt engine file located in ./models',
                        required=False)
    parser.add_argument('-i', '--image', default='100.png', help='image file path', required=False)
    args = parser.parse_args()
    return {'model': args.model, 'image': args.image}


def resize_pos(boxes, src_size, tar_size):
    w1 = src_size[0]
    h1 = src_size[1]
    w2 = tar_size[0]
    h2 = tar_size[1]
    # y2 = (h2 / h1) * y1
    # x2 = (w2 / w1) * x1

    boxes[:, [0, 2]] = (w2 / w1) * boxes[:, [0, 2]]
    boxes[:, [1, 3]] = (h2 / h1) * boxes[:, [1, 3]]

    # coords[:, [0, 2]] -= pad[0]  # x padding
    # coords[:, [1, 3]] -= pad[1]  # y padding
    return boxes

def main():
    # parse arguments
    args = cli()

    # setup processor and visualizer
    processor = Processor(model=args['model'])
    visualizer = Visualizer()

    # fetch input
    print('image arg', args['image'])
    img = cv2.imread('inputs/{}'.format(args['image']))
    im0 = img.copy()
    img = processor.pre_process_my(img)

    output, fps = processor.detect(img)
    img = cv2.resize(im0, (672, 672))

    # final results
    boxes, confs, classes = processor.post_process(output)

    # Rescale boxes from img_size to im0 size
    # boxes = processor.scale_coords(img.shape[1:], boxes, im0.shape).round()
    # boxes = resize_pos(boxes,(672,672),im0.shape)
    res = visualizer.draw_results(img, boxes, confs, classes)
    cv2.imwrite("aa.jpg", res)


if __name__ == '__main__':
    main()
