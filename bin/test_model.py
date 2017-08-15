#! /usr/bin/env python
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s [%(levelname)s] : %(message)s')
logger = logging.getLogger('test_model')

def prepare_output_file(filename):
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e

if __name__ == '__main__':
    main_parser = argparse.ArgumentParser(description='Train models for classification of chest X-ray radiography')
    subparsers = main_parser.add_subparsers(dest='command')

    parser = subparsers.add_parser('classify_types',
                                   help='a simple classifier for types')
    parser.add_argument('-i', '--input-file', type=str, required=True,
                        help='input HDF5 file')
    parser.add_argument('--model-file', type=str, required=True)
    parser.add_argument('--indices-file', type=str,
                        help='HDF5 file containing training indices')
    parser.add_argument('--indices-name', type=str,
                        help='HDF5 dataset name of training indices')
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('-o', '--output-file', type=str, required=True,
                        help='output model file')
    args = main_parser.parse_args()

    if args.command == 'classify_types':
        import numpy as np
        import h5py
        from keras.models import load_model
        from sklearn.metrics import accuracy_score, roc_auc_score

        logger.info('read input file: ' + args.input_file)
        fin = h5py.File(args.input_file, 'r')
        X = fin['X'][:]
        positions = fin['position'][:]
        y = (positions == 'C').astype('int32')
        fin.close()

        if args.indices_file is not None:
            fin = h5py.File(args.indices_file, 'r')
            indices = fin[args.indices_name][:]
            fin.close()
            X = np.take(X, indices, axis=0)
            y = np.take(y, indices, axis=0)
            logger.info('number of test samples: %d'%indices.shape[0])
        X = np.repeat(X, 3, axis=3)

        logger.info('load model: ' + args.model_file)
        model = load_model(args.model_file)
        y_pred = np.ravel(model.predict(X))
        y_pred_labels = (y_pred > 0.5).astype('int32')
        accuracy = accuracy_score(y, y_pred_labels)
        roc_auc = roc_auc_score(y, y_pred)
        logger.info('accuracy: %f'%(accuracy))
        logger.info('roc_auc: %f' % (roc_auc))

        logger.info('save results to file: ' + args.output_file)
        prepare_output_file(args.output_file)
        fout = h5py.File(args.output_file, 'w')
        fout.create_dataset('y_true', data=y)
        fout.create_dataset('y_pred', data=y_pred)
        fout.create_dataset('y_pred_labels', data=y_pred_labels)
        g = fout.create_group('metrics')
        g.create_dataset('accuracy', data=accuracy)
        g.create_dataset('roc_auc', data=roc_auc)
        fout.close()

