import cv2
import sys
import argparse

from Processor import Processor
from Visualizer import Visualizer
import os

img_dir = "/home/user/Code/uva/multi_detection/data/images/visible"


def cli():
    desc = 'Run TensorRT yolov5 visualizer'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-m', '--model', default='yolov5s-simple-32.trt', help='trt engine file located in ./models',
                        required=False)
    parser.add_argument('-i', '--image', default='1239.png', help='image file path', required=False)
    args = parser.parse_args()
    return {'model': args.model, 'image': args.image}


def main():
    # parse arguments
    args = cli()

    # setup processor and visualizer
    processor = Processor(model=args['model'])
    visualizer = Visualizer()

    # fetch input
    FPS = 0
    for im in os.listdir(img_dir):
        img = cv2.imread(os.path.join(img_dir, im))

        # inference
        output, fps = processor.detect(img)
        FPS += fps
        img = cv2.resize(img, (672, 672))

        # final results
        boxes, confs, classes = processor.post_process(output)
        visualizer.save_results(img, boxes, confs, classes,im)
    print("avg fps:",FPS/len(os.listdir(img_dir)))

if __name__ == '__main__':
    main()
