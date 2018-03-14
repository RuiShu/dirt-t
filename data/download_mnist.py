import os
import numpy as np
import subprocess
from scipy.io import loadmat, savemat
from skimage.transform import resize

def mnist_resize(x):
    H, W, C = 32, 32, 3
    x = x.reshape(-1, 28, 28)
    resized_x = np.empty((len(x), H, W), dtype='float32')
    for i, img in enumerate(x):
        # resize returns [0, 1]
        resized_x[i] = resize(img, (H, W), mode='reflect')

    # Retile to make RGB
    resized_x = resized_x.reshape(-1, H, W, 1)
    resized_x = np.tile(resized_x, (1, 1, 1, C))
    return resized_x

def main():
    if os.path.exists('mnist.npz'):
        print "Using existing mnist.npz"

    else:
        print "Opening subprocess to download data from URL"
        subprocess.check_output(
            '''
            wget https://s3.amazonaws.com/img-datasets/mnist.npz
            ''',
            shell=True)

    if os.path.exists('mnist32_train.mat') and os.path.exists('mnist32_test.mat'):
        print "Using existing mnist32_train.mat and mnist32_test.mat"

    else:
        print "Resizing mnist.npz to (32, 32, 3)"
        data = np.load('mnist.npz')
        trainx = data['x_train']
        trainy = data['y_train']
        trainx = mnist_resize(trainx)
        savemat('mnist32_train.mat', {'X': trainx, 'y': trainy})

        testx = data['x_test']
        testy = data['y_test']
        testx = mnist_resize(testx)
        savemat('mnist32_test.mat', {'X': testx, 'y': testy})

    print "Loading mnist32_train.mat for sanity check"
    data = loadmat('mnist32_train.mat')
    print data['X'].shape, data['X'].min() ,data['X'].max()
    print data['y'].shape, data['y'].min() ,data['y'].max()

    print "Loading mnist32_test.mat for sanity check"
    data = loadmat('mnist32_test.mat')
    print data['X'].shape, data['X'].min() ,data['X'].max()
    print data['y'].shape, data['y'].min() ,data['y'].max()


if __name__ == '__main__':
    main()
