import keras
import numpy as np
import os
import scipy.ndimage
import random
import cv2
import keras.backend as K


def load_image(image_index, image_base_path, size):
    batch_value = np.zeros(size)
    for i in image_index:
        image_path = os.path.join(image_base_path, "{}.jpg.npy".format(i))
        image = np.load(image_path)
        batch_value[i] = image
    return batch_value


#model.fit_generator(generate_arrays_from_file('./my_file.txt'),samples_per_epoch=10000,nb_epoch=10)

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def std_evaluate(model, generator, size):
    """
    """
    #size = generator.get_size()
    #batch_size = generator.get_batch_size()
    #n_batches = size // batch_size
    print("std test")

    err_sum = 0.
    err_count = 0.
    count = 0
    for data in generator:
        count += 1
        X_batch, y_batch = data
        y_pred = model.predict_on_batch(X_batch)
        err_sum += np.sum((y_batch - y_pred) ** 2)
        err_count += len(y_pred)
        if count == size:
            break

    mse = err_sum / err_count
    return [mse, np.sqrt(mse)]

def std_evaluate_seq(model, generator, size, seq_size):
    """
    """
    #size = generator.get_size()
    #batch_size = generator.get_batch_size()
    #n_batches = size // batch_size
    print("std test")

    err_sum = 0.
    err_count = 0.
    count = 0
    for data in generator:
        count += 1
        X_batch, y_batch = data
        y_pred = model.predict_on_batch(X_batch)
        err_sum += np.sum((y_batch - y_pred) ** 2)
        err_count += len(y_pred)
        if count == size:
            break

    mse = err_sum / err_count
    return [mse, np.sqrt(mse)]

#https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1


def trans_image(image, steer, trans_range):
    rows, cols, chan = image.shape
    # Translation
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    steer_ang = steer + tr_x / trans_range * 2 * .2
    tr_y = 40 * np.random.uniform() - 40 / 2
    # tr_y = 0
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, Trans_M, (cols, rows))

    return image_tr, steer_ang


def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image

def generate_arrays_from_file_new(labels, index_values, image_path_base, batch_size, scale=1.0, random_flip=False, input_shape=(120, 320, 3)):
    batch_features = np.zeros((batch_size, *input_shape))
    batch_labels = np.zeros((batch_size, 1))
    while True:
        next_indexes = np.random.choice(np.arange(0, len(index_values)), batch_size)
        for i, idx in enumerate(next_indexes):
            #idx = np.random.choice(len(labels), 1)
            y = labels[idx]
            image_path = os.path.join(image_path_base, "{}.jpg.npy".format(int(index_values[idx])))
            image = np.load(image_path)
            if random_flip:
                flip_bit = random.randint(0, 1)
                if flip_bit == 1:
                    image = np.flip(image, 1)
                    y = y * -1
            image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
            image = ((image-(255.0/2))/255.0)
            batch_features[i, :] = image
            batch_labels[i] = y * scale
        yield batch_features, batch_labels
        #f.close()

def generate_arrays_from_file_new_all_cam(labels, index_values, image_path_base, batch_size, scale=1.0, random_flip=False, input_shape=(120, 320, 3)):
    batch_features = np.zeros((batch_size, *input_shape))
    batch_labels = np.zeros((batch_size, 1))
    while True:
        cam_id = random.randint(0, 2)
        if cam_id == 0:
            next_indexes = np.random.choice(np.arange(0, len(index_values[cam_id])), batch_size)
        elif cam_id == 1:
            next_indexes = np.random.choice(np.arange(0, len(index_values[cam_id])), batch_size)
        elif cam_id == 2:
            next_indexes = np.random.choice(np.arange(0, len(index_values[cam_id])), batch_size)


        for i, idx in enumerate(next_indexes):
            #idx = np.random.choice(len(labels), 1)
            y = labels[cam_id][idx]
            image_path = os.path.join(image_path_base[cam_id], "{}.jpg.npy".format(int(index_values[cam_id][idx])))
            image = np.load(image_path)
            if random_flip:
                flip_bit = random.randint(0, 1)
                if flip_bit == 1:
                    image = np.flip(image, 1)
                    y = y * -1
            image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
            image = ((image-(255.0/2))/255.0)
            batch_features[i, :] = image
            batch_labels[i] = y * scale
        yield batch_features, batch_labels
        #f.close()


def generate_arrays_from_file_new_augment(labels, index_values, image_path_base, batch_size, scale=1.0):
    batch_features = np.zeros((batch_size, 120, 320, 3))
    batch_labels = np.zeros((batch_size, 1))
    while True:
        next_indexes = np.random.choice(np.arange(0, len(index_values)), batch_size)
        for i, idx in enumerate(next_indexes):
            y = labels[idx]
            image_path = os.path.join(image_path_base, "{}.jpg.npy".format(int(index_values[idx])))
            image = np.load(image_path)
            flip_bit = random.randint(0, 1)
            if flip_bit == 1:
                image = np.flip(image, 1)
                y = y * -1

            #image, y = trans_image(image, y, 150)
            image = add_random_shadow(image)
            image = augment_brightness_camera_images(image)
            #image = scipy.ndimage.interpolation.rotate(image, random.uniform(-15, 15), reshape=False)

            image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
            image = ((image-(255.0/2))/255.0)
            batch_features[i, :] = image
            batch_labels[i] = y * scale
        yield batch_features, batch_labels
        #f.close()

def generate_arrays_from_file_new_augment_light(labels, index_values, image_path_base, batch_size, scale=1.0):
    batch_features = np.zeros((batch_size, 120, 320, 3))
    batch_labels = np.zeros((batch_size, 1))
    while True:
        next_indexes = np.random.choice(np.arange(0, len(index_values)), batch_size)
        for i, idx in enumerate(next_indexes):
            y = labels[idx]
            image_path = os.path.join(image_path_base, "{}.jpg.npy".format(int(index_values[idx])))
            image = np.load(image_path)
            flip_bit = random.randint(0, 1)
            if flip_bit == 1:
                image = np.flip(image, 1)
                y = y * -1

            #image, y = trans_image(image, y, 150)
            #image = add_random_shadow(image)
            image = augment_brightness_camera_images(image)
            #image = scipy.ndimage.interpolation.rotate(image, random.uniform(-15, 15), reshape=False)

            image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
            image = ((image-(255.0/2))/255.0)
            batch_features[i, :] = image
            batch_labels[i] = y * scale
        yield batch_features, batch_labels
        #f.close()

def generate_arrays_from_file_new_augment_aggressive(labels, index_values, image_path_base, batch_size, scale=1.0):
    batch_features = np.zeros((batch_size, 120, 320, 3))
    batch_labels = np.zeros((batch_size, 1))
    while True:
        for i in range(batch_size):
            #idx = np.random.choice(len(labels), 1)
            leave_loop = False
            while leave_loop==False:
                idx = np.random.choice(len(labels), 1)
                y = labels[idx]
                if abs(y) < 0.15:
                    leave_prob = np.random.uniform()
                    if leave_prob > 0.9:
                        leave_loop = True
                else:
                    leave_loop = True
            y = labels[idx]
            image_path = os.path.join(image_path_base, "{}.jpg.npy".format(int(index_values[idx])))
            image = np.load(image_path)
            flip_bit = random.randint(0, 1)
            if flip_bit == 1:
                image = np.flip(image, 1)
                y = y * -1

            image, y = trans_image(image, y, 150)
            image = add_random_shadow(image)
            image = augment_brightness_camera_images(image)
            image = scipy.ndimage.interpolation.rotate(image, random.uniform(-45, 45), reshape=False)

            image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
            image = ((image-(255.0/2))/255.0)
            batch_features[i, :] = image
            batch_labels[i] = y * scale
        yield batch_features, batch_labels
        #f.close()


def generate_arrays_from_file_new_3d(labels, index_values, image_path_base, batch_size, scale=1.0, number_of_frames=1, random_flip=False, input_shape=(120,320,3)):
    batch_features = np.zeros((batch_size, number_of_frames, *input_shape))
    batch_labels = np.zeros((batch_size, 1))
    value_range = np.arange(0,len(labels)-number_of_frames-1)
    while True:
        next_indexes = np.random.choice(np.arange(0, len(index_values) - number_of_frames - 1), batch_size)
        for i, idx in enumerate(next_indexes):
            for j in range(number_of_frames):
                y = labels[idx+j]
                image_path = os.path.join(image_path_base, "{}.jpg.npy".format(int(index_values[idx+j])))
                image = np.load(image_path)
                if random_flip:
                    flip_bit = random.randint(0, 1)
                    if flip_bit == 1:
                        image = np.flip(image, 1)
                        y = y * -1
                image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
                image = ((image - (255.0 / 2)) / 255.0)
                batch_features[i, j, :] = image
                batch_labels[i] = y * scale


        yield batch_features, batch_labels


def generate_arrays_from_file_new_3d_seq(labels, index_values, image_path_base, batch_size, scale=1.0, number_of_frames=1, seq_length=1, random_flip=False):
    batch_features = np.zeros((batch_size, seq_length, number_of_frames, 120, 320, 3))
    batch_labels = np.zeros((batch_size, seq_length, 1))
    value_range = np.arange(0, len(labels)-number_of_frames-seq_length-1)
    while True:
        next_indexes = np.random.choice(value_range, batch_size)
        for batch_i, idx in enumerate(next_indexes):
            for seq in range(seq_length):
                for frame in range(number_of_frames):
                    y = labels[idx + frame + seq]
                    image_path = os.path.join(image_path_base, "{}.jpg.npy".format(int(index_values[idx + frame + seq])))
                    image = np.load(image_path)
                    if random_flip:
                        flip_bit = random.randint(0, 1)
                        if flip_bit == 1:
                            image = np.flip(image, 1)
                            y = y * -1
                    image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
                    image = ((image - (255.0 / 2)) / 255.0)
                    batch_features[batch_i, seq, frame, :] = image
                    batch_labels[batch_i, seq, :] = y * scale

        yield batch_features, batch_labels



def generate_arrays_from_file_new_3d_with_diff(labels, index_values, image_path_base, batch_size, scale=1.0, number_of_frames=1):
    batch_features = np.zeros((batch_size, number_of_frames, 120, 320, 3))
    batch_labels = np.zeros((batch_size, 1))
    value_range = np.arange(1,len(labels)-number_of_frames-1)
    while True:
        for i in range(batch_size):
            idx = np.random.choice(value_range, 1)
            for j in range(number_of_frames):
                y = labels[idx+j]
                y_prev = labels[idx + j -1]
                image_path = os.path.join(image_path_base, "{}.jpg.npy".format(int(index_values[idx])))
                image = np.load(image_path)
                image_path_prev = os.path.join(image_path_base, "{}.jpg.npy".format(int(index_values[idx-1])))
                image_prev = np.load(image_path_prev)
                flip_bit = random.randint(0, 1)
                if flip_bit == 1:
                    image = np.flip(image, 1)
                    y = y * -1
                    image_prev = np.flip(image_prev, 1)
                    y_prev = y_prev * -1
                image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
                image = ((image - (255.0 / 2)) / 255.0)

                image_prev[:, :, 0] = cv2.equalizeHist(image_prev[:, :, 0])
                image_prev = ((image_prev - (255.0 / 2)) / 255.0)

                batch_features[i, j, :] = image - image_prev
                batch_labels[i] = y * scale


        yield batch_features, batch_labels
        #f.close()



def image_convert(image):
    image_out = image* 255.0 + 255.0/2
    image_out = cv2.cvtColor(image_out, cv2.COLOR_YUV2BGR)
    return image_out


def get_images_single(start_image_path, start):
    data = np.zeros((1, 160, 320, 3))
    image_copies = np.zeros((1, 160, 320, 3))
    image_path = os.path.join(start_image_path, '{}.jpg'.format(start))
    image = cv2.imread(image_path, )
    image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
    image = image.astype('uint8')
    image_copy = np.copy(image)
    image_copies[0, :] = image_copy
    image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
    x = ((image - (255.0 / 2)) / 255.0)
    data[0, :] = x

    return data, image_copies



def get_images(start_image_path, number_of_frames, start):
    data = np.zeros((1, number_of_frames, 120, 320, 3))
    image_copies = np.zeros((1, number_of_frames, 120, 320, 3))
    for i in range(number_of_frames):
        image_path = os.path.join(start_image_path, '{}.jpg.npy'.format(start+i))
        image = np.load(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
        image = image.astype('uint8')
        image_copy = np.copy(image)
        image_copies[0, i, :] = image_copy
        image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
        x = ((image - (255.0 / 2)) / 255.0)
        data[0, i, :] = x
    return data, image_copies

def get_images_seq(start_image_path, number_of_frames, seq_length, start):
    data = np.zeros((1, seq_length, number_of_frames, 120, 320, 3))
    image_copies = np.zeros((1, seq_length, number_of_frames, 120, 320, 3))
    for seq in range(seq_length):
        for i in range(number_of_frames):
            image_path = os.path.join(start_image_path, '{}.jpg.npy'.format(start + i + seq))
            image = np.load(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
            image = image.astype('uint8')
            image_copy = np.copy(image)
            image_copies[0, i, :] = image_copy
            image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
            x = ((image - (255.0 / 2)) / 255.0)
            data[0, seq, i, :] = x

    return data, image_copies

def get_images_single_res(start_image_path, start):
    data = np.zeros((1, 224, 224, 3))
    image_copies = np.zeros((1, 224, 224, 3))
    image_path = os.path.join(start_image_path, '{}.jpg.npy'.format(start))
    image = np.load(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
    image = image.astype('uint8')
    image_copy = np.copy(image)
    image_copies[0, :] = image_copy
    image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
    x = ((image - (255.0 / 2)) / 255.0)
    data[0, :] = x

    return data, image_copies

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * np.math.pow(drop, np.math.floor((1 + epoch) / epochs_drop))
    return lrate
