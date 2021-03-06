from __future__ import absolute_import, division, print_function

import face_recognition

"""

This is a modification of the classify_images.py
script in Tensorflow. The original script produces
string labels for input images (e.g. you input a picture
of a cat and the script returns the string "cat"); this
modification reads in a directory of images and
generates a vector representation of the image using
the penultimate layer of neural network weights.

It is a modified version of the script found here:
https://douglasduhaime.com/posts/identifying-similar-images-with-tensorflow.html

This one is made to work with Flask and in accordance to the requirements set by Sørlandets kunstmuseum.
"""

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple image classification with Inception.

Run image classification with Inception trained on ImageNet 2012 Challenge data
set.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.

https://tensorflow.org/tutorials/image_recognition/
"""

import glob
import json
import os
import os.path
import pathlib
import re
import sys
import tarfile
from collections import defaultdict
import cv2
import numpy as np
import psutil
import tensorflow.compat.v1 as tf
from annoy import AnnoyIndex
from flask import Flask, request, redirect, render_template, jsonify
from scipy import spatial
from six.moves import urllib
from werkzeug.utils import secure_filename
from timeit import default_timer as timer
import sys

sys.argv = sys.argv[:1]

tf.config.optimizer.set_jit(True)
tf.compat.v1.enable_eager_execution()

tf.app.flags.DEFINE_string('bind', '', 'Server address')
tf.app.flags.DEFINE_integer('timeout', 30, 'Server timeout')
tf.app.flags.DEFINE_string('host', '', 'Server host')
tf.app.flags.DEFINE_integer('port', 80, 'Server port')

FLAGS = tf.app.flags.FLAGS

# classify_image_graph_def.pb:
#   Binary representation of the GraphDef protocol buffer.
# imagenet_synset_to_human_label_map.txt:
#   Map from synset ID to a human readable string.
# imagenet_2012_challenge_label_map_proto.pbtxt:
#   Text representation of a protocol buffer mapping a label to synset ID.
tf.app.flags.DEFINE_string(
    'model_dir', '/tmp/imagenet',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")
tf.app.flags.DEFINE_string('image_file', '',
                           """Absolute path to image file.""")
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'


# pylint: enable=line-too-long


class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""

    def __init__(self,
                 label_lookup_path=None,
                 uid_lookup_path=None):
        if not label_lookup_path:
            label_lookup_path = os.path.join(
                FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
        if not uid_lookup_path:
            uid_lookup_path = os.path.join(
                FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        """Loads a human readable English name for each softmax node.
        Args:
          label_lookup_path: string UID to integer node ID.
          uid_lookup_path: string UID to human-readable string.
        Returns:
          dict from integer node ID to human-readable string.
        """
        if not tf.io.gfile.exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.io.gfile.exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)

        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.io.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string

        # Loads mapping from string UID to integer node ID.
        node_id_to_uid = {}
        proto_as_ascii = tf.io.gfile.GFile(label_lookup_path).readlines()
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        # Loads the final mapping of integer node ID to human-readable string
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
                tf.logging.fatal('Failed to locate: %s', val)
            name = uid_to_human[val]
            node_id_to_name[key] = name

        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]


def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.io.gfile.GFile(os.path.join(
            FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def run_inference_on_images(image_list, output_dir):
    """Runs inference on an image list.
    Args:
      image_list: a list of images.
      output_dir: the directory in which image vectors will be saved
    Returns:
      image_to_labels: a dictionary with image file keys and predicted
        text label values
    """
    image_to_labels = defaultdict(list)

    create_graph()

    with tf.compat.v1.Session() as sess:
        # Some useful tensors:
        # 'softmax:0': A tensor containing the normalized prediction across
        #   1000 labels.
        # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
        #   float description of the image.
        # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
        #   encoding of the image.
        # Runs the softmax tensor by feeding the image_data as input to the graph.
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')

        for image_index, image in enumerate(image_list):
            try:
                print("parsing", image_index, image, "\n")
                if not tf.io.gfile.exists(image):
                    tf.logging.fatal('File does not exist %s', image)

                with tf.io.gfile.GFile(image, 'rb') as f:
                    image_data = f.read()

                    process = psutil.Process(os.getpid())
                    mem3 = process.memory_info().rss
                    print('Memory After reading file', mem3 / (1024 ** 2), 'MB')

                    predictions = sess.run(softmax_tensor,
                                           {'DecodeJpeg/contents:0': image_data})

                    predictions = np.squeeze(predictions)

                    ###
                    # Get penultimate layer weights
                    ###

                    feature_tensor = sess.graph.get_tensor_by_name('pool_3:0')
                    feature_set = sess.run(feature_tensor,
                                           {'DecodeJpeg/contents:0': image_data})
                    feature_vector = np.squeeze(feature_set)
                    outfile_name = os.path.basename(image) + ".npz"
                    out_path = os.path.join(output_dir, outfile_name)
                    np.savetxt(out_path, feature_vector, delimiter=',')

                    # Creates node ID --> English string lookup.
                    node_lookup = NodeLookup()

                    process = psutil.Process(os.getpid())
                    mem4 = process.memory_info().rss
                    print('Memory before prediction', mem4 / (1024 ** 2), 'MB')

                    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
                    for node_id in top_k:
                        human_string = node_lookup.id_to_string(node_id)
                        score = predictions[node_id]
                        print("results for", image)
                        print('%s (score = %.5f)' % (human_string, score))
                        print("\n")

                        image_to_labels['image_labels'].append(
                            {
                                "labels": human_string,
                                "score": str(score)
                            }
                        )
                process = psutil.Process(os.getpid())
                mem5 = process.memory_info().rss
                print('Memory After Prediction', mem5 / (1024 ** 2), 'MB')

                # close the open file handlers
                proc = psutil.Process()
                open_files = proc.open_files()

                for open_file in open_files:
                    file_handler = getattr(open_file, "fd")
                    os.close(file_handler)
            except Exception as e:
              s = str(e)
              print('could not process image index', image_index, 'image', image)

    return image_to_labels


def maybe_download_and_extract():
    """Download and extract model tar file."""
    dest_directory = FLAGS.model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def run_classify_images(original_file_path, output_dir):
    maybe_download_and_extract()
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    images = glob.glob(original_file_path)

    image_to_labels = run_inference_on_images(images, output_dir)

    # detect number of faces
    num_faces = face_recognition.detect_num_faces(original_file_path)
    image_to_labels['number_of_faces'].append(num_faces)

    process = psutil.Process(os.getpid())
    mem6 = process.memory_info().rss
    print('Memory After Face Detection', mem6 / (1024 ** 2), 'MB')

    image_to_labels_path = os.path.join(os.getcwd(), 'tmp', 'image_to_labels.json')
    with open(image_to_labels_path, "w") as img_to_labels_out:
        json.dump(image_to_labels, img_to_labels_out)

    print("calssification done")
