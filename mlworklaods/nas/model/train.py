import json
import numpy as np
import model_spec
# Specify the file path
input_file_path = "mlworklaods/nas/model/generated_graphs.json"
available_ops = ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3']
max_edges = 9
# Open the file in read mode and use json.load to load the data into a Python object
with open(input_file_path, 'r') as f:
    models = json.load(f)


ordered_keys = sorted(models.keys())
# 'loaded_buckets' now contains the Python object from the JSON file

for model_id in ordered_keys:
    matrix, labels = models[model_id]
    print(matrix)
    matrix = np.array(matrix)
    labels = (['input'] +[available_ops[lab] for lab in labels[1:-1]] +['output'])
    spec = model_spec.ModelSpec(matrix, labels)
    assert spec.valid_spec


