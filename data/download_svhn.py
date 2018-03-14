import subprocess
import os
from scipy.io import loadmat 

def main():
    if os.path.exists('test_32x32.mat') and os.path.exists('train_32x32.mat'):
        print "Using existing data"

    else:
        print "Opening subprocess to download data from URL"
        subprocess.check_output(
            '''
            wget http://ufldl.stanford.edu/housenumbers/train_32x32.mat
            wget http://ufldl.stanford.edu/housenumbers/test_32x32.mat
            wget http://ufldl.stanford.edu/housenumbers/extra_32x32.mat
            ''',
            shell=True)

    print "Loading train_32x32.mat for sanity check"
    data = loadmat('train_32x32.mat')
    print data['X'].shape, data['X'].min() ,data['X'].max()
    print data['y'].shape, data['y'].min() ,data['y'].max()

    print "Loading test_32x32.mat for sanity check"
    data = loadmat('test_32x32.mat')
    print data['X'].shape, data['X'].min() ,data['X'].max()
    print data['y'].shape, data['y'].min() ,data['y'].max()


if __name__ == '__main__':
    main()
