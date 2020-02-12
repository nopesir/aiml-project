import os

import cv2
import numpy as np
from keras.models import load_model, Input, Model
from keras.layers import Conv2D
from keras.utils import plot_model
from matplotlib import pyplot as plt, cm

import util
from vis.visualization import visualize_saliency

model = load_model('../model-002.h5')
#plot_model(model, to_file="model.png", show_shapes=True, dpi=200)
#model = model_from_json(open('nvidia_no_aug.h5').read())
#model.load_weights(os.path.join(os.path.dirname('nvidia_no_aug'), 'model_weights.h5'))

#model.layers.pop(0)
#inp = Input(batch_shape=(160, 320, 3))
#newInput = Conv2D(3, (5, 5), strides=2, activation='relu', input_shape=(1,160, 320,3))(inp) # let us say this new InputLayer
#newOutputs = model(newInput)
#model = Model(inp, newOutputs)


model.summary()
for idx, layer in enumerate(model.layers):
    print(idx, layer)

#data, image_copies = util.get_images_single_res("M:\\selfdrive\\SelfDrivingData\\test_out3\\training\\images\\center\\", 10000)
#data, image_copies = util.get_images_single("", "center_2019_04_02_19_28_48_991")
#nvidia 11
#res 6

a = "1475522435613512147"
layer = 10
img=cv2.imread( str(a)+ ".jpg", cv2.z)
#img, image_copies = util.get_images_single_res("", "center_2019_04_02_19_28_48_991")
print(img.shape)
image = img[120:-50, :, :] 
image = cv2.resize(img, (200, 66), cv2.INTER_AREA)
print(image.shape)
image = np.expand_dims(image, axis=0)
print(image.shape)
pred_value = model.predict(image)
heatmap, grads = visualize_saliency(model, layer, [0], image)

print(grads.shape)
grads_new = np.sum(grads, axis=0)
heatmap_new = np.uint8(cm.jet(grads_new)[..., :3] * 255)
heatmap_new_no_aug = np.uint8(image[0] * .5 + heatmap_new * (1. - .5))

plt.imshow(heatmap_new_no_aug)
plt.savefig("yes" + str(layer) + "-" + a + ".jpg", bbox_inches="tight")
plt.show()

plt.imshow(img)
plt.savefig("no-" + a + ".jpg", bbox_inches="tight")
plt.show()


print('Model loaded.')
#
# # The name of the layer we want to visualize
# # (see model definition in vggnet.py)
# layer_name = 'predictions'
# layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]
#
#





