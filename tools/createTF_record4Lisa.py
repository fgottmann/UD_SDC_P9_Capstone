import tensorflow as tf
import os
import io
import csv
import PIL.Image
from lxml import etree
from object_detection.utils import dataset_util, label_map_util
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import yaml

flags = tf.app.flags

flags.DEFINE_string('data_dir', None, 'Path to the folder where the images are stored')
flags.DEFINE_string('labels_name', None, 'Filename of label file')
flags.DEFINE_string('labels_map_path', None, 'Path to the labels map pbtxt file')
flags.DEFINE_float('split_train_test', 0.25, 'If supplied specifies the amount of samples to use for evaluation')

tf.app.flags.mark_flag_as_required('data_dir')
tf.app.flags.mark_flag_as_required('labels_name')
tf.app.flags.mark_flag_as_required('labels_map_path')

FLAGS = flags.FLAGS

def create_tf_example(data_dir, label, labels_map):
    file_path = os.path.join(data_dir, label['Filename'].split('/')[-1])
    print(file_path)
    print()
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

    width = int(1280)
    height = int(960)

    filename = label['Filename'].encode('utf8')
    ignore = 0 # ignore unknown situations

    xmin.append(float(label['Upper left corner X']) / width)
    ymin.append(float(label['Upper left corner Y']) / height)
    xmax.append(float(label['Lower right corner X']) / width)
    ymax.append(float(label['Lower right corner Y']) / height)
    if label['Annotation tag'] == 'warning':
        label['Annotation tag'] = 'Yellow'
    elif label['Annotation tag'] == 'stop':
        label['Annotation tag'] = 'Red'
    elif label['Annotation tag'] == 'go':
        label['Annotation tag'] = 'Green'
    elif label['Annotation tag'] == 'goLeft':
        label['Annotation tag'] = 'GreenLeft'
    elif label['Annotation tag'] == 'stopLeft':
        label['Annotation tag'] = 'RedLeft'
    elif label['Annotation tag'] == 'warningLeft':
        label['Annotation tag'] = 'Yellow'
    elif label['Annotation tag'] == 'goRight':
        label['Annotation tag'] = 'GreenRight'
    elif label['Annotation tag'] == 'stopRight':
        label['Annotation tag'] = 'RedRight'
    elif label['Annotation tag'] == 'warningRight':
        label['Annotation tag'] = 'Yellow'
    elif label['Annotation tag'] == 'goForward':
        label['Annotation tag'] = 'GreenStraight'
    elif label['Annotation tag'] == 'stopForward':
        label['Annotation tag'] = 'RedStraight'
    elif label['Annotation tag'] == 'warningForward':
        label['Annotation tag'] = 'Yellow'

    if 0:
        if label['Annotation tag'] == 'GreenRight':
            label['Annotation tag'] = 'Green'
            ignore = 1
        elif label['Annotation tag'] == 'RedRight':
            label['Annotation tag'] = 'Red'
            ignore = 1
        elif label['Annotation tag'] == 'GreenLeft':
            label['Annotation tag'] = 'Green'
            ignore = 1
        elif label['Annotation tag'] == 'RedLeft':
            label['Annotation tag'] = 'Red'
            ignore = 1
        elif label['Annotation tag'] == 'GreenStraight':
            label['Annotation tag'] = 'Green'
            ignore = 1
        elif label['Annotation tag'] == 'RedStraight':
            label['Annotation tag'] = 'Red'
            ignore = 1
        elif label['Annotation tag'] == 'GreenStraightLeft':
            label['Annotation tag'] = 'Green'
            ignore = 1
        elif label['Annotation tag'] == 'RedStraightLeft':
            label['Annotation tag'] = 'Red'
            ignore = 1
        elif label['Annotation tag'] == 'GreenStraightRight':
            label['Annotation tag'] = 'Green'
            ignore = 1
        elif label['Annotation tag'] == 'RedStraightRight':
            label['Annotation tag'] = 'Red'
            ignore = 1
        elif label['Annotation tag'] == 'off':
            a = 0
            #label['Annotation tag'] = 'Red'
            #ignore = 1

    classes_text.append(label['Annotation tag'].encode('utf8'))
    classes.append(labels_map[label['Annotation tag']])
    truncated.append(int(0))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpg'.encode('utf8')),
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

def handle_directory(data_dir,labels_name, labels_map, output_path,output_name,split_train_test):

    elements = os.listdir(data_dir)
    if labels_name in elements:
        data_dir_intern = os.path.join(data_dir,'frames');
        if os.path.isdir(data_dir_intern):
            labels_train = []
            reader = csv.DictReader(open(os.path.join(data_dir,labels_name),'rt'), delimiter=";")
            for row in reader:
                labels_train.append(row)
            if split_train_test:
                labels_train, labels_eval = train_test_split(labels_train, test_size = split_train_test, shuffle = True)

            output_path_train = os.path.join(output_path, '{}_train.record'.format(output_name))

            create_tf_record(labels_train, data_dir_intern, labels_map, output_path_train)
            print('TF record file for training created with {} samples: {}'.format(len(labels_train), output_path_train))

            if labels_eval:
                output_path_eval = os.path.join(output_path, '{}_eval.record'.format(output_name))
                create_tf_record(labels_eval, data_dir_intern, labels_map, output_path_eval)
                print('TF record file for validation created with {} samples: {}'.format(len(labels_eval), output_path_eval))

    else:
        for el in elements:
            new_path = os.path.join(data_dir,el);
            new_name = "{}_{}".format(output_name,el)
            if os.path.isdir(new_path):
                handle_directory(new_path,labels_name, labels_map, output_path,new_name,split_train_test)

def main(unused_argv):

    data_dir = FLAGS.data_dir
    labels_map = label_map_util.get_label_map_dict(FLAGS.labels_map_path)
    labels_name = FLAGS.labels_name
    output_path = data_dir
    split_train_test = FLAGS.split_train_test
    output_name = 'dataset_lisa'


    handle_directory(data_dir,labels_name, labels_map, output_path,output_name,split_train_test)


if __name__ == '__main__':
  tf.app.run()
