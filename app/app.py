from __future__ import absolute_import, division, print_function
from google.cloud import error_reporting

import json
import os
import os.path
import pathlib
import psutil
import uuid
from flask import Flask, request, redirect, render_template, jsonify
from werkzeug.utils import secure_filename
from timeit import default_timer as timer
import cv2
import classify_images
import cluster_vectors

app = Flask(__name__, template_folder='template')
app.secret_key = "v9y/B?E(H+MbQeTh"
error_client = error_reporting.Client()
keepImages = False

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

def resize_image(original_file_path):
    img = cv2.imread(original_file_path, cv2.IMREAD_UNCHANGED)
    scale_percent = 45000/img.shape[0]  # around 400px height
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite(original_file_path, resized)
    return

#@app.route('/api/stockimage', methods=['POST'])
#def stock_image():
#  global keepImages
#  keepImages = True
#  return process_image(request)

@app.route("/api/userimage", methods=['POST'])
def user_image():
  global keepImages
  keepImages = False
  return process_image(request)

def process_image(request):
  global keepImages
  if request.method == 'POST':
    try:
      process = psutil.Process(os.getpid())
      mem0 = process.memory_info().rss
      print('Memory Usage Before Action', mem0 / (1024 ** 2), 'MB')
      start = timer()

      temp_dir = os.getcwd() + '/tmp'
      if not os.path.isdir(temp_dir):
        os.makedirs(temp_dir)

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
          newname = uuid.uuid4().hex + '.' + file.filename.split('.')[1]
          original_filename = secure_filename(newname)
          original_file_path = os.path.join(temp_dir, original_filename)
          # file.save(original_filename)
          file.save(original_file_path)
          print('Saved upload to file: ', original_file_path)

      resize_image(original_file_path)
      classify_images.run_classify_images(original_file_path, temp_dir)
      cluster_vectors.cluster_vectors(original_file_path)

      mem11 = process.memory_info().rss
      print('Memory After Classification and Clustering', mem11 / (1024 ** 2), 'MB')

      nn_file_path = os.path.join(os.getcwd(), 'tmp', 'nearest_neighbors', original_filename.split('.')[0] + '.json')
      with open(nn_file_path) as nn_file:
          data = json.load(nn_file)
      image_to_labels_path = os.path.join(os.getcwd(), 'app', 'image_to_labels.json')
      with open(image_to_labels_path, "rb") as itl_file:
          labels = json.load(itl_file)
      output_list = []
      output_list.append(data)
      output_list.append(labels)
      #output_list = [data, labels]
      npz_file_path = temp_dir + '/' + original_filename + '.npz'
      if not keepImages:
        os.remove(npz_file_path)
        os.remove(nn_file_path)
        os.remove(original_file_path)

      end = timer()
      print('Total time = ' + str(end - start))  # Time in seconds, e.g. 5.38091952400282
      mem1 = process.memory_info().rss
      print('Memory Usage After Action', mem1 / (1024 ** 2), 'MB')
      print('Memory Increase After Action', (mem1 - mem0) / (1024 ** 2), 'MB')

      return jsonify(output_list)
    except Exception as e:
      s = str(e)
      print(s)
      error_client.report_exception()

@app.route('/result/<result>')
def result_string(result):
    return 'Result: %s!' % result


@app.route('/neighbours', methods=['GET'])
def neighbours():
    return render_template("project_1.html")


if __name__ == '__main__':
    app.run(debug=True)
#app.run(host='0.0.0.0', port=80, debug=True)
