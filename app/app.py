from __future__ import absolute_import, division, print_function

import json
import os
import os.path
import pathlib
import psutil
from flask import Flask, request, redirect, render_template, jsonify
from werkzeug.utils import secure_filename
from timeit import default_timer as timer

import classify_images
import cluster_vectors

app = Flask(__name__, template_folder='template')
app.secret_key = "v9y/B?E(H+MbQeTh"

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
            classify_images.run_classify_images(image_name)
            cluster_vectors.cluster_vectors(image_name)

            mem11 = process.memory_info().rss
            print('Memory After Classification and Clustering', mem11 / (1024 ** 2), 'MB')
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


@app.route('/neighbours', methods=['GET'])
def neighbours():
    return render_template("project_1.html")


if __name__ == '__main__':
    app.run(debug=True)
# app.run(host='0.0.0.0', port=80, debug=True)
