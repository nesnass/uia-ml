from __future__ import absolute_import, division, print_function

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
import tensorflow as tf
from annoy import AnnoyIndex
from flask import Flask, request, redirect, render_template, jsonify
from scipy import spatial
from six.moves import urllib
from werkzeug.utils import secure_filename
from timeit import default_timer as timer

tf.config.optimizer.set_jit(True)
tf.compat.v1.enable_eager_execution()

app = Flask(__name__, template_folder='template')
app.secret_key = "v9y/B?E(H+MbQeTh"


def cluster_vectors(name):
    # data structures
    file_index_to_file_name = {}
    file_index_to_file_vector = {}

    # config
    dims = 2048
    n_nearest_neighbors = 30
    trees = 10000
    infiles = glob.glob('static/image_vectors/*.npz')

    # build ann index
    t = AnnoyIndex(dims)
    for file_index, i in enumerate(infiles):
        file_vector = np.loadtxt(i)
        file_name = os.path.basename(i).split('.')[0]
        file_index_to_file_name[file_index] = file_name
        file_index_to_file_vector[file_index] = file_vector
        t.add_item(file_index, file_vector)
    t.build(trees)

    # create a nearest neighbors json file for each input
    if not os.path.exists('static/nearest_neighbors'):
        os.makedirs('static/nearest_neighbors')
    i = [num for num, val in file_index_to_file_name.items() if val == name.split('.')[0]][0]
    # for i in file_index_to_file_name.keys():
    master_file_name = file_index_to_file_name[i]
    master_vector = file_index_to_file_vector[i]

    named_nearest_neighbors = []
    nearest_neighbors = t.get_nns_by_item(i, n_nearest_neighbors)
    for j in nearest_neighbors:
        neighbor_file_name = file_index_to_file_name[j]
        neighbor_file_vector = file_index_to_file_vector[j]

        similarity = 1 - spatial.distance.cosine(master_vector, neighbor_file_vector)
        rounded_similarity = int((similarity * 10000)) / 10000.0

        named_nearest_neighbors.append({
            'filename': neighbor_file_name,
            'similarity': rounded_similarity
        })

    with open('static/nearest_neighbors/' + master_file_name + '.json', 'w') as out:
        json.dump(named_nearest_neighbors, out)


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


def detect_num_faces(image):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    process = psutil.Process(os.getpid())
    mem30 = process.memory_info().rss
    print('Memory During Face Detection', mem30 / (1024 ** 2), 'MB')
    return len(faces)


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

                # # detect number of faces
                num_faces = detect_num_faces(image)
                image_to_labels['number_of_faces'].append(num_faces)

                process = psutil.Process(os.getpid())
                mem6 = process.memory_info().rss
                print('Memory After Face Detection', mem6 / (1024 ** 2), 'MB')

                # close the open file handlers
                proc = psutil.Process()
                open_files = proc.open_files()

                for open_file in open_files:
                    file_handler = getattr(open_file, "fd")
                    os.close(file_handler)
            except:
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


def run_classify_images(image_name):
    maybe_download_and_extract()
    output_dir = "static/image_vectors"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    images = glob.glob(image_name)

    image_to_labels = run_inference_on_images(images, output_dir)

    with open("image_to_labels.json", "w") as img_to_labels_out:
        json.dump(image_to_labels, img_to_labels_out)

    print("all done")


@app.route('/', methods=['GET'])
def hello():
    return "Hello World!"


@app.route('/upload', methods=['GET'])
def upload():
    return render_template("file_upload_form.html")


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/api", methods=['POST'])
def api():
    if request.method == 'POST':
        process = psutil.Process(os.getpid())
        mem0 = process.memory_info().rss
        print('Memory Usage Before Action', mem0 / (1024 ** 2), 'MB')
        start = timer()

        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(filename)
        image_name = filename
        image_output = 'static/image_vectors/' + image_name + '.npz'
        file = pathlib.Path(image_output)
        if not file.exists():
            run_classify_images(image_name)

            mem10 = process.memory_info().rss
            print('Memory After Classification', mem10 / (1024 ** 2), 'MB')

            cluster_vectors(image_name)

            mem11 = process.memory_info().rss
            print('Memory After Clustering', mem11 / (1024 ** 2), 'MB')

            os.remove(filename)
        s = 'static/nearest_neighbors/' + image_name.split('.')[0] + '.json'
        output_list = []
        with open(s) as json_file:
            data = json.load(json_file)
        with open('image_to_labels.json', "rb") as infile:
            labels = json.load(infile)
        output_list.append(data)
        output_list.append(labels)
        os.remove(image_output)
        os.remove(s)

        end = timer()
        print('Total time = ' + str(end - start))  # Time in seconds, e.g. 5.38091952400282
        mem1 = process.memory_info().rss
        print('Memory Usage After Action', mem1 / (1024 ** 2), 'MB')
        print('Memory Increase After Action', (mem1 - mem0) / (1024 ** 2), 'MB')

        return jsonify(output_list)


@app.route('/result/<result>')
def result_string(result):
    return 'Result: %s!' % result


if __name__ == '__main__':
    app.run(debug=True)
