from annoy import AnnoyIndex
from scipy import spatial
import json, glob, os
import numpy as np

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

  # create a nearest neighbors json file for the image
  if not os.path.exists('static/nearest_neighbors'):
    os.makedirs('static/nearest_neighbors')
  i = [num for num, val in file_index_to_file_name.items() if val == name.split('.')[0]][0]
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
