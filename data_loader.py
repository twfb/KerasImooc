from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
path = 'E:\\GOD\Documents\\VsCode\\PyNoteBook\\Keras\\'


def get_train_data(cat_or_dog, start, end):
    x = np.zeros([1, 64, 64, 3])
    y = np.zeros([1, 1])
    
    for i in range(start, end + 1):
        image = '{}images\\train\\{}.{}.jpg'.format(path, cat_or_dog, i)
        label = [1]
        if cat_or_dog == 'dog':
            label = [0]
        image = np.array(plt.imread(image))
        num_px = 64
        image = resize(image, (num_px, num_px))
        x = np.insert(x, 0, values=image, axis=0)
        y = np.insert(y, 0, values=label, axis=0)
    
    x = np.delete(x, -1, axis = 0)
    y = np.delete(y, -1, axis = 0)
    return x, y


def save_data(cat_or_dog, train_num, test_num):
    x_train, y_train = get_train_data(cat_or_dog, 0, train_num)
    x_test, y_test = get_train_data(cat_or_dog, 10000,10000 -1 + test_num)
    np.savez('{}{}'.format(path, cat_or_dog), x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)


# save_data('cat', 200, 20)
# save_data('dog', 200, 20)


def load_cat_or_dog_data(cat_or_dog):
    data = np.load('{}.npz'.format(cat_or_dog))
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    return (x_train, y_train), (x_test, y_test)

def load_data():
    (x_train_cat, y_train_cat), (x_test_cat, y_test_cat) = load_cat_or_dog_data('cat')
    (x_train_dog, y_train_dog), (x_test_dog, y_test_dog) = load_cat_or_dog_data('dog')
    x_train = np.append(x_train_cat, x_train_dog, axis=0)
    y_train = np.append(y_train_cat, y_train_dog, axis=0)
    x_test = np.append(x_test_cat, x_test_dog, axis=0)
    y_test = np.append(y_test_cat, y_test_dog, axis=0)
    permutation_train = np.random.permutation(y_train.shape[0])
    permutation_test = np.random.permutation(y_test.shape[0])
    x_train = x_train[permutation_train, :, :]
    y_train = y_train[permutation_train]
    x_test = x_train[permutation_test, :, :]
    y_test = y_train[permutation_test]
    return (x_train, y_train), (x_test, y_test)
