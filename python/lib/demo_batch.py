import cv2
import sys
import argparse

from ProcessorMy import Processor
from Visualizer import Visualizer
import os
import time
batch_img = "/home/user/Code/uva/multi_detection/data/images/visible"

save_txt = "/home/user/Code/yolov5-tensorrt-master/python/lib/output/save_txt"
save_img = "/home/user/Code/yolov5-tensorrt-master/python/lib/output/save_img"


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
    boxes[:, [0, 2]] = (w2 / w1) * boxes[:, [0, 2]]
    boxes[:, [1, 3]] = (h2 / h1) * boxes[:, [1, 3]]
    return boxes


def main():
    # parse arguments
    args = cli()

    # setup processor and visualizer
    processor = Processor(model=args['model'])
    visualizer = Visualizer()

    set_img = os.listdir(batch_img)
    for im in set_img:
        # fetch input
        print('image arg', im)
        img = cv2.imread(os.path.join(batch_img, im))
        h, w, _ = img.shape
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
        cv2.imwrite(os.path.join(save_img, im), img)
        del output, img,img_rs
        # time.sleep(1)



if __name__ == '__main__':
    main()
