from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np

def get_train_data(cat_or_dog, start, end):
    x = np.zeros([1, 64, 64, 3])
    y = np.zeros([1, 1])
    
    for i in range(start, end + 1):
        image = 'E:\\GOD\Documents\\VsCode\\PyNoteBook\\Keras\\images\\train\\{}.{}.jpg'.format(cat_or_dog, i)
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
    x_test, y_test =get_train_data(cat_or_dog, 10000,10000 -1 + test_num)
    np.savez('E:\\GOD\Documents\\VsCode\\PyNoteBook\\Keras\\{}'.format(cat_or_dog), x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)


# save_data('cat', 3000, 200)
# save_data('dog', 3000, 200)


def load_data(cat_or_dog):
    data = np.load('E:\\GOD\Documents\\VsCode\\PyNoteBook\\Keras\\{}.npz'.format(cat_or_dog))
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    return (x_train, y_train), (x_test, y_test)