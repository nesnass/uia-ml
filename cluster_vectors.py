from annoy import AnnoyIndex
from scipy import spatial
import json, glob, os
import numpy as np

# data structures
file_index_to_file_name = {}
file_index_to_file_vector = {}

# stock_vectors are the source image vectors we are comparing the new image to
stock_vector_path = os.path.join(os.getcwd(), 'static', 'image_vectors')
stock_vectors = glob.glob(stock_vector_path + '/*.npz')
for index, item in enumerate(stock_vectors):
  file_vector = np.loadtxt(item)
  file_name = os.path.basename(item).split('.')[0]
  file_index_to_file_name[index] = file_name
  file_index_to_file_vector[index] = file_vector

userImageIndex = index + 1

def cluster_vectors(original_file_path):

  # config
  dims = 2048
  n_nearest_neighbors = 30
  trees = 10000

  # include the new user image as the final item (overwrite previous if necessary)
  file_basename = os.path.basename(original_file_path)
  npz_file_path = os.path.join(os.getcwd(), 'tmp', file_basename + '.npz')
  master_vector = np.loadtxt(npz_file_path)
  master_file_name = os.path.basename(original_file_path).split('.')[0]
  file_index_to_file_name[userImageIndex] = master_file_name
  file_index_to_file_vector[userImageIndex] = master_vector

  # build an index of source images
  t = AnnoyIndex(dims, 'angular')
  for key in file_index_to_file_vector:
    t.add_item(key, file_index_to_file_vector[key])

  # build trees
  t.build(trees)

  # create a nearest neighbors json file for the image
  nnpath = os.path.join(os.getcwd(), 'tmp', 'nearest_neighbors')
  if not os.path.exists(nnpath):
    os.makedirs(nnpath)
  # i = [num for num, val in file_index_to_file_name.items() if val == filename.split('.')[0]][0]
  #master_file_name = file_index_to_file_name[index]
  #master_vector = file_index_to_file_vector[index]

  named_nearest_neighbors = []
  nearest_neighbors = t.get_nns_by_item(index, n_nearest_neighbors)
  for j in nearest_neighbors:
    neighbor_file_name = file_index_to_file_name[j]
    neighbor_file_vector = file_index_to_file_vector[j]

    similarity = 1 - spatial.distance.cosine(master_vector, neighbor_file_vector)
    rounded_similarity = int((similarity * 10000)) / 10000.0

    named_nearest_neighbors.append({
      'filename': neighbor_file_name,
      'similarity': rounded_similarity
    })

  json_output_file = os.path.join(nnpath, master_file_name + '.json')
  with open(json_output_file, 'w') as out:
    json.dump(named_nearest_neighbors, out)
  t.unload()
  print("clustering done")
