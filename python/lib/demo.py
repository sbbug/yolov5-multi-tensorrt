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
    parser.add_argument('-i', '--image', default='605.png', help='image file path', required=False)
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
    h, w, _ = img.shape
    im0 = img.copy()
    im0 = cv2.resize(im0, (672, 672))
    print(img.shape)
    # pre-process
    img_rs = processor.pre_process_my(img)
    # detect
    output, fps = processor.detect(img_rs)
    # post-process
    output = processor.post_process(output)

    # output[:, :4] = processor.scale_coords(img_rs.shape[1:], output[:, :4], img0.shape).round()
    output = resize_pos(output, (672, 672), (w, h))

    visualizer.draw_box_save(output, img)
    cv2.imwrite("605_res-2.jpg", img)
    # visualizer.draw_results(img, boxes, confs, classes, args['image'])


if __name__ == '__main__':
    main()
