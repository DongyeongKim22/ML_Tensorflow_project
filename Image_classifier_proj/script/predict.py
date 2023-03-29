# copy some lines from previous workspce and do it
# Import TensorFlow 
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

# Ignore some warnings that are not relevant (you can remove this if you prefer)
import warnings
warnings.filterwarnings('ignore')
# Some other recommended settings:
tfds.disable_progress_bar()
# TODO: Make all other necessary imports.
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import json
import time
import numpy as np
import argparse
from PIL import Image

#parser setting
parser = argparse.ArgumentParser(description = 'setting image path, model name, and so on',)
parser.add_argument('arg1', action = 'store', help = 'image path')
parser.add_argument('arg2', action = 'store', help = 'model name')
parser.add_argument('--top_k', type = int, default = 5, help = 'top class number')
parser.add_argument('--category_names',action = 'store' , default = 'map.json', help = 'name of category')

args = parser.parse_args()
#image size defin
image_size = 224

json_name = 'label_{}'.format(args.category_names)
with open(json_name, 'r') as f:
    class_names = json.load(f)

print(len(class_names))


# TODO: Load the Keras model

saved_keras_model_filepath = './{}'.format(args.arg2)
reload_savedModel = tf.keras.models.load_model(saved_keras_model_filepath, custom_objects={'KerasLayer':hub.KerasLayer})
reload_savedModel.summary()



# TODO: Create the process_image function
def process_image(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    image = image.numpy()
    return image

# TODO: Create the predict function
def predict(image_path, model, top_k):
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)
    image_extended = np.expand_dims(image, axis = 0)
    probes = model.predict(image_extended)
    
    top_probs, labels = tf.math.top_k(probes, k=top_k)
    
    top_probs = top_probs.numpy()
    labels = labels.numpy()

    
    return top_probs, labels, image

# TODO: Plot the input image along with the top 5 classes




url = '{}'.format(args.arg1)
top_probes, labels, image = predict(url, reload_savedModel, args.top_k)
print('Propabilties:', top_probes)
print('labels :', labels)

classes = []
for i in labels[0]:
    classes.append(class_names[str(i)])
    print(classes)
    
fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
ax1.imshow(image, cmap = plt.cm.binary)
ax1.axis('off')
ax1.set_title(classes[0])
ax2.barh(np.arange(5),top_probes[0])
ax2.set_aspect(0.1)
ax2.set_yticks(np.arange(5))
ax2.set_yticklabels(classes);
ax2.set_title('Class Probability')
ax2.set_xlim(0, 1.1)
plt.tight_layout()
plt.show()