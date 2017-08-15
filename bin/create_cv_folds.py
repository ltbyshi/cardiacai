#! /usr/bin/env python
import argparse, os, sys, errno
import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('create_cv_folds')

def prepare_output_file(filename):
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n-samples', type=int, required=True,
        help='number of samples')
    parser.add_argument('-o', '--output-file', type=str, required=True,
        help='prefix for output file names')
    parser.add_argument('-k', type=int, default=5)
    args = parser.parse_args()

    from sklearn.model_selection import KFold
    import numpy as np
    import h5py

    prepare_output_file(args.output_file)
    fout = h5py.File(args.output_file, 'w')
    i = 0
    kfold = KFold(args.k, shuffle=True)
    for train_index, test_index in kfold.split(np.arange(args.n_samples)):
        g = fout.create_group(str(i))
        g.create_dataset('train', data=train_index)
        g.create_dataset('test', data=test_index)
        i += 1
    fout.close()