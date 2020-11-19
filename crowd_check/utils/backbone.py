import glob, os
from utils import label_map_util
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def set_model(model_name, label_name):
	# model_found = 0

	# for file in glob.glob("*"):
	# if (file == model_name):
	# model_found = 1

	# Path to frozen detection graph. This is the actual model that is used for the object detection.
	path_to_ckpt = model_name + '/frozen_inference_graph.pb'

	# List of the strings that is used to add correct label for each box.
	path_to_labels = os.path.join('data', label_name)

	num_classes = 90


# Load a (frozen) Tensorflow model into memory.
	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')

# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts 5, we know that this corresponds to airplane. Here I 		use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
	label_map = label_map_util.load_labelmap(path_to_labels)
	categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
	category_index = label_map_util.create_category_index(categories)

	return detection_graph, category_index
