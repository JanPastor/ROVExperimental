import os
import cv2
import numpy as np
import tensorflow as tf
from midasnet.keras_layers import midasnet
from midasnet.utils import convert_midas_weights
from tensorflow.contrib import layers
from tf_object_detection.utils import label_map_util
from tf_object_detection.utils import visualization_utils as vis_util 

# Set Parameters
MODEL_NAME = 'ssd_inception_v2_coco_2018_01_28'
LABEL_NAME = 'mscoco_label_map.pbtxt'

# Set File Paths 
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', LABEL_NAME)

# Load pretrained model and running inference 
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session(graph=detection_graph)

# Define the depth estimation graph 
def midasnet_model():
    cam = layers.conv2d(inputs, 1, (3, 3), padding='SAME', activation_fn=None)

    # Residual Block 1 
    residual1 = midasnet.MidasResidualBlock(cam, num_layers=2, out_channels=32,
                                            is_training=True)
    pool1 = layers.max_pool2d(residual1, (2, 2))

    # Residual Block 2 
    residual2 = midasnet.MidasResidualBlock(pool1, num_layers=3, out_channels=32,
                                            is_training=True)
    pool2 = layers.max_pool2d(residual2, (2, 2))

    # Residual Block 3 
    residual3 = midasnet.MidasResidualBlock(pool2, num_layers=4, out_channels=32,
                                            is_training=True)
    pool3 = layers.max_pool2d(residual3, (2, 2))

    # Midas Fusion Layer
    fusion = midasnet.MidasFuseLayer(pool3, out_channels=32, is_training=True)

    # Deconvolution Layer 
    output = midasnet.MidasDeconvolutionLayer(fusion, out_channels=1, is_training=True)

    return output 
        
# Load label map 
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)
        
# Read in video 
cap = cv2.VideoCapture(os.path.join(CWD_PATH, 'vid.mp4'))

# Width and height of the video 
width = int(cap.get(3))
height = int(cap.get(4))

while True:
    
    ret, frame = cap.read()
    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the object detection 
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # Find only boxes with score greater than the threshold 
    boxes_out = []
    scores_out = []
    classes_out = []
    
    for i in range(len(scores[0])):
        if scores[0][i] > 0.5:
            # Get the box, class, score data 
            box = boxes[0][i]
            classes = classes[0][i]
            score = scores[0][i]
            class_name = category_index[classes]['name']

            # Draw the box, class, score data on the frame
            vis_util.draw_bounding_box_on_image_array(
                frame,
                box,
                score,
                classes,
                min_score_thresh=0.5,
                use_normalized_coordinates=True,
                line_thickness=2)

            # Store the box, class, score data 
            boxes_out.append(box)
            scores_out.append(score)
            classes_out.append(class_name)
    
    # Display the results 
    cv2.imshow('Object Detection', frame)

    # Run the depth estimation graph 
    depth_estimation = midasnet_model()
    
    # Calculate the depth of each detected object
    for i in range(len(boxes_out)):
        # Crop image inside the detected object
        ymin, xmin, ymax, xmax = boxes_out[i]
        xmin = int(xmin*width)
        xmax = int(xmax*width)
        ymin = int(ymin*height)
        ymax = int(ymax*height)
        obj_img = frame[ymin:ymax, xmin:xmax]

        # Calculate the depth of the detected object 
        output_depth = sess.run(depth_estimation, feed_dict={inputs: obj_img})
        depth_val = output_depth[0, 0] * 65535.0 + 5000.0
        
        # Display the depth of the detected object 
        cv2.putText(frame, 'Est.Depth: {}'.format(depth_val), (xmin, ymin),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Exit the program with the "q" key 
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break