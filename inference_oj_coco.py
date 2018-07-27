import numpy as np 
import os 
import tensorflow as tf
from collections import defaultdict
from PIL import Image
from matplotlib import pyplot as plt

ckpt_path = "./"
ckpt_file = "ssd_coco_frozen.pb"


class ObjectDetection():
	
	def __init__(self, model_ckpt, class_names_dict):
		self.model_ckpt = model_ckpt
		self.graph = None
		self.class_names = np.load(class_names_dict).item()	
	
		
	def load_model(self):
		detection_graph = tf.Graph()
		with detection_graph.as_default():
			graph_def = tf.GraphDef()
			with tf.gfile.GFile(self.model_ckpt, 'rb') as fid:
				serialized_graph = fid.read()
				graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(graph_def, name='')
		self.graph = detection_graph
		
	def load_image_into_numpy_array(self, image):
		
	  (im_width, im_height) = image.size	  
	  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
	
	def prepare_image(self, image_path):  
		image = Image.open(image_path)
		image = self.load_image_into_numpy_array(image)
		image = np.expand_dims(image, axis =0)		
		return image
		

	def run_detection_inference(self, image, conf_thresh):
		with self.graph.as_default():
			
			with tf.Session() as sess:
				
				operations = tf.get_default_graph().get_operations()
				layer_names = {output.name for op in operations for output in op.outputs}     
				
				layer_dict = {}
				
				model_output_keys = ['num_detections', 
									'detection_boxes', 
									'detection_scores', 
									'detection_classes', 
									]
				
				for key in model_output_keys:
					layer_name = key + ':0'				
					if layer_name in layer_names:
						layer_dict[key] = tf.get_default_graph().get_tensor_by_name(layer_name)
				
				image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')				
				
				output = sess.run(layer_dict, feed_dict={image_tensor:image})
				
				boxes = np.transpose(output["detection_boxes"], [1,2,0])
				boxes = boxes.reshape(boxes.shape[0], boxes.shape[1])
				classid = np.transpose(output["detection_classes"])
				scores = np.transpose(output["detection_scores"])
				detections = np.concatenate([classid, scores, boxes ], axis = 1)
				
				num_filtered_detection = np.sum([1 for score in scores if score > conf_thresh])				
				detections = detections[0:num_filtered_detection]
				
			return detections
	
	def write_image(self,image_path, detections, output="output.jpg"):
		
		img = Image.open(image_path)
		img = self.load_image_into_numpy_array(img)
		currentAxis = plt.gca()
		num_classes = len(self.class_names)
		colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()
		color_code = min(len(detections), 16)
		
		height = img.shape[0]
		width = img.shape[1]


		for detection in detections:
		    # Parse the outputs.

		    det_label = detection[0]
		    det_conf = detection[1]
		    det_ymin = detection[2]*height
		    det_xmin = detection[3]*width
		    det_ymax = detection[4]*height
		    det_xmax = detection[5]*width
		  
		    xmin = int(det_xmin)
		    ymin = int(det_ymin)
		    xmax = int(det_xmax)
		    ymax = int(det_ymax)
		    score = det_conf
		    label = int(det_label)
		    label_name = self.class_names[label]
		    
		    print label_name
		    plt.imshow(img / 255.)
		    
		    # label_name = class_names[label]
		    # print label_name 
		    # print label

		    # display_txt = '{:0.2f}, {}'.format(score, label_name)
		    display_txt = '{}'.format(label_name)
		    
		    coords = (xmin, ymin), (xmax-xmin), (ymax-ymin)
		    color_code = color_code-1 
		    
		    color = colors[color_code]
		    currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=1))
		    currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.70})
		    # currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.70}, fontsize =7)
		plt.savefig(output)

		plt.clf()		


conf_thresh = 0.3
image_path = "./"
class_names_dict = "coco_class_names.npy"
# image_name = "000000000016.jpg"
# image_name = "000000000019.jpg"
image_name = "image1.jpg"
# image_name = "000000000063.jpg"
# image_name = "000000000069.jpg"

# image_name = "000000000016.jpg"


objdet = ObjectDetection(ckpt_path + ckpt_file, class_names_dict)
objdet.load_model()


image = objdet.prepare_image(image_path + image_name)
detections = objdet.run_detection_inference(image, conf_thresh)
objdet.write_image(image_path+image_name, detections, "outdoor_out.jpg")

	



