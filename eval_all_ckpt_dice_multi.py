import cv2
import time
import math
import os
import glob
import numpy as np
import tensorflow as tf

import lanms

tf.app.flags.DEFINE_string('test_data_path', '../data/Oxford/test/', '')#''#../data/Oxford/ori/test_dataset/test_data/images/
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', '../model/hand_resnet_v1_50_dice_multi_oxford_aug_rbox/', '')
tf.app.flags.DEFINE_string('output_dir', '../result/oxford-test-result/pos/', '')#'/tmp/ch4_test_images/images/'
tf.app.flags.DEFINE_string('heatmap_output_dir', '../result/oxford-test-result/heatmaps_resenet_dice_multi_oxford/', '')
tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')

import resnet_v1_model_dice_multi as model
from oxford import restore_rectangle

FLAGS = tf.app.flags.FLAGS
CKPT_PATH = None#'../model/hand_resnet_v1_50_dice_multi_oxford_aug_rbox/model.ckpt-98051'#
heatmaps = []
MAX_SIDE_LEN = 2400#400#512#
THRESHOLD = 0.1


def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for ext in exts:
        files += glob.glob(os.path.join(FLAGS.test_data_path, '*.' + ext))
    print('Find {} images'.format(len(files)))
    return files


def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    # h, w, _ = im.shape
    #
    # resize_w = w
    # resize_h = h
    resize_h, resize_w = h, w = im.shape[:2]
    # h, w = im.shape[:2]
    # resize_h, resize_w = int(h*1.5), int(w*1.5)#512, 512#


    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
    #     ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    # else:
    #     ratio = 1.
    # resize_h = int(resize_h * ratio)
    # resize_w = int(resize_w * ratio)
        ratio = float(max_side_len) / max(resize_w, resize_h)
        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)
    #
    # resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    # resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    resize_h = ((resize_h + 3) // 32) * 32
    resize_w = ((resize_w + 3) // 32) * 32

    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    # ratio_h = resize_h / float(h)
    # ratio_w = resize_w / float(w)
    #
    # return im, (ratio_h, ratio_w)
    return im, (resize_h / float(h), resize_w / float(w))


def detect(im_fn, im, ratio_h, ratio_w, score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    '''
    restore hand boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    # global heatmaps
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # updated
    # resize_score_map = cv2.resize(score_map, (score_map.shape[1]*4, score_map.shape[0]*4))
    # cv2.imwrite(FLAGS.heatmap_output_dir+os.path.basename(im_fn)[:-4]+'_heatmap.png', resize_score_map*255)
    
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the hand boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    print('{} hand boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]

    if not FLAGS.no_write_images:
        im_copy = im[:, :, ::-1].copy()
        boxes_temp = boxes.copy()
        if boxes is not None:
            confidence = boxes_temp[:, 8].reshape(-1, 1)
            boxes_temp = boxes_temp[:, :8].reshape((-1, 4, 2))
            boxes_temp[:, :, 0] /= ratio_w
            boxes_temp[:, :, 1] /= ratio_h
        if boxes_temp is not None:
            for box, c in zip(boxes_temp,confidence):
                box = sort_poly(box.astype(np.int32))
                cv2.polylines(im_copy, [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
        cv2.imwrite(os.path.join(FLAGS.heatmap_output_dir,
                    os.path.basename(im_fn)[:-4]+'_with_bboxes_before_nms.jpg'), im_copy)

    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def main(argv=None):
    sum_time_dic = {}
    avg_time_dic = {}
    global heatmaps
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list


    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise
    try:
        os.makedirs(FLAGS.heatmap_output_dir)
    except OSError as e:
        if e.errno != 17:
            raise

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        f_score, f_geometry = model.model(input_images, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        with tf.Session(config=config) as sess:
            checkpoints = [FLAGS.checkpoint_path+'model.ckpt-116060.index']#glob.glob(FLAGS.checkpoint_path+'model.ckpt-2*index')
            checkpoints.sort()
            for checkpoint in checkpoints:#
                sum_time = 0
                checkpoint_path = checkpoint[:-6]

                global_step = checkpoint_path.split('/')[-1].split('-')[-1]
                print('Succesfully loaded model from %s at step=%s.' %
                    (checkpoint_path, global_step))
                saver.restore(sess, checkpoint_path)

                im_fn_list = get_images()
                with open(FLAGS.output_dir+'for-images-oxford-'+str(MAX_SIDE_LEN)+'-dice-multi-aug-' +global_step+'.txt', 'w') as f:
                    ii = 0
                    for im_fn in im_fn_list:
                        ii += 1
                        # if os.path.exists(FLAGS.output_dir+os.path.basename(im_fn)[:-4]+'.txt'):
                        #     continue
                        im = cv2.imread(im_fn)[:, :, ::-1]
                        start_time = time.time()
                        im_resized, (ratio_h, ratio_w) = resize_image(im, MAX_SIDE_LEN)

                        timer = {'net': 0, 'restore': 0, 'nms': 0}
                        start = time.time()
                        score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
                        timer['net'] = time.time() - start

                        #updated
                        boxes, timer = detect(im_fn, im, ratio_h, ratio_w, score_map=score[3], geo_map=geometry[3], timer=timer, box_thresh=THRESHOLD)
                        print('{}({}) : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
                            im_fn, ii, timer['net']*1000, timer['restore']*1000, timer['nms']*1000))

                        if boxes is not None:
                            confidence = boxes[:, 8].reshape(-1, 1)
                            boxes = boxes[:, :8].reshape((-1, 4, 2))
                            boxes[:, :, 0] /= ratio_w
                            boxes[:, :, 1] /= ratio_h


                        duration = time.time() - start_time
                        print('[timing] {}'.format(duration))
                        sum_time += duration

                        # save to file
                        if boxes is not None:
                            # res_file = os.path.join(
                            #     FLAGS.output_dir,
                            #     '{}.txt'.format(
                            #         os.path.basename(im_fn).split('.')[0]))

                            # with open(res_file, 'w') as f:
                            for box, c in zip(boxes,confidence):
                                # to avoid submitting errors
                                box = sort_poly(box.astype(np.int32))
                                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                                    continue
                                f.write(os.path.basename(im_fn)+',{},{},{},{},{},{},{},{},{}\r\n'.format(
                                    box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1], round(float(c), 6)
                                ))
                                cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
                        else:
                            print('None')
                        if not FLAGS.no_write_images:
                            img_path = os.path.join(FLAGS.heatmap_output_dir, os.path.basename(im_fn))
                            cv2.imwrite(img_path, im[:, :, ::-1])
                    sum_time_dic[int(global_step)] = sum_time
                    avg_time_dic[int(global_step)] = float(sum_time)/ii
            print('----------sum_time----------')
            print(sum_time_dic)
            print('----------avg_time----------')
            print(avg_time_dic)
            print('-----------------------------')
            print('avg_time:', sum(avg_time_dic.values())/len(avg_time_dic.values()))

if __name__ == '__main__':
    tf.app.run()
