import tensorflow as tf
import os
import io
import PIL.Image
from lxml import etree
from object_detection.utils import dataset_util, label_map_util
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import yaml

flags = tf.app.flags

flags.DEFINE_string('data_dir', None, 'Path to the folder where the images are stored')
flags.DEFINE_string('labels_path', None, 'Path to the file in which labels are stored')
flags.DEFINE_string('labels_map_path', None, 'Path to the labels map pbtxt file')
flags.DEFINE_string('output_path', None, 'Path to output record file, if split_train_test is enabled creates two file one for training and one for validation')
flags.DEFINE_float('split_train_test', 0.25, 'If supplied specifies the amount of samples to use for evaluation')

tf.app.flags.mark_flag_as_required('data_dir')
tf.app.flags.mark_flag_as_required('labels_path')
tf.app.flags.mark_flag_as_required('labels_map_path')

FLAGS = flags.FLAGS

def create_tf_example(data_dir, label, labels_map):
    print(label['path'])
    file_path = os.path.join(data_dir, label['path'])
    with tf.gfile.GFile(file_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)

    if image.format != 'JPEG':
        imagename = file_path.split('/')[-1]
        if imagename[-3:] == 'png':
            im = PIL.Image.open(file_path)
            temp_file = os.path.join(data_dir, 'temp.jpg')
            im.save(temp_file)

            with tf.gfile.GFile(temp_file, 'rb') as fid:
                encoded_jpg = fid.read()
            encoded_jpg_io = io.BytesIO(encoded_jpg)
            image = PIL.Image.open(encoded_jpg_io)

    if image.format != 'JPEG':
        raise ValueError('Image format not JPG')

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []

    widthreal = int(1280)
    heightreal = int(720)

    width = int(1280)
    height = int(720)

    filename = label['path'].encode('utf8')
    ignore = 0 # ignore unknown situations
    for box in label['boxes']:
        xmin.append(float(box['x_min']) / width)
        ymin.append(float(box['y_min']) / height)
        xmax.append(float(box['x_max']) / width)
        ymax.append(float(box['y_max']) / height)
        if 1:
            if box['label'] == 'GreenRight':
                box['label'] = 'Green'
                #ignore = 1
            elif box['label'] == 'RedRight':
                box['label'] = 'Red'
                #ignore = 1
            elif box['label'] == 'GreenLeft':
                box['label'] = 'Green'
                #ignore = 1
            elif box['label'] == 'RedLeft':
                box['label'] = 'Red'
                #ignore = 1
            elif box['label'] == 'GreenStraight':
                box['label'] = 'Green'
                #ignore = 1
            elif box['label'] == 'RedStraight':
                box['label'] = 'Red'
                #ignore = 1
            elif box['label'] == 'GreenStraightLeft':
                box['label'] = 'Green'
                #ignore = 1
            elif box['label'] == 'RedStraightLeft':
                box['label'] = 'Red'
                #ignore = 1
            elif box['label'] == 'GreenStraightRight':
                box['label'] = 'Green'
                #ignore = 1
            elif box['label'] == 'RedStraightRight':
                box['label'] = 'Red'
                #ignore = 1
            elif box['label'] == 'off':
                a = 0
                #box['label'] = 'Red'
                #ignore = 1

        classes_text.append(box['label'].encode('utf8'))
        classes.append(labels_map[box['label']])
        truncated.append(int(box['occluded']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(r'jpg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/truncated': dataset_util.int64_list_feature(truncated)
    }))
    return tf_example, ignore

def create_tf_record(label_files, data_dir, labels_map, output_path):

    writer = tf.python_io.TFRecordWriter(output_path)

    for label in tqdm(label_files, desc='Converting', unit=' images'):
        tf_record, ignore = create_tf_example(data_dir, label, labels_map)
        if ignore == 0:
            writer.write(tf_record.SerializeToString())

    writer.close()

def main(unused_argv):

    data_dir = FLAGS.data_dir
    labels_map = label_map_util.get_label_map_dict(FLAGS.labels_map_path)
    label_file = FLAGS.labels_path
    if FLAGS.output_path is None:
        output_path_train = data_dir
    else:
        output_path_train = FLAGS.output_path
    split_train_test = FLAGS.split_train_test


    labels_train = yaml.load(open(label_file,'rb').read(),Loader=yaml.FullLoader)
    print('Total samples: {}'.format(len(labels_train)))

    if split_train_test:
        labels_train, labels_eval = train_test_split(labels_train, test_size = split_train_test, shuffle = True)
        dir_path = os.path.dirname(output_path_train)

        if len(dir_path) and not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        file_name_split = os.path.splitext(os.path.basename(output_path_train))

        if file_name_split[1] == '':
            file_name_split = (file_name_split[0], '.record')

        output_path_train = os.path.join(dir_path, '{}_train{}'.format(file_name_split[0], file_name_split[1]))
        output_path_eval = os.path.join(dir_path, '{}_eval{}'.format(file_name_split[0], file_name_split[1]))


    create_tf_record(labels_train, data_dir, labels_map, output_path_train)
    print('TF record file for training created with {} samples: {}'.format(len(labels_train), output_path_train))

    if labels_eval:
        create_tf_record(labels_eval, data_dir, labels_map, output_path_eval)
        print('TF record file for validation created with {} samples: {}'.format(len(labels_eval), output_path_eval))

if __name__ == '__main__':
  tf.app.run()
