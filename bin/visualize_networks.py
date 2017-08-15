#! /usr/bin/env python
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s [%(levelname)s] : %(message)s')
logger = logging.getLogger('visualize_networks')

def prepare_output_file(filename):
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e

if __name__ == '__main__':
    main_parser = argparse.ArgumentParser(description='Train models for classification of chest X-ray radiography')
    subparsers = main_parser.add_subparsers(dest='command')

    parser = subparsers.add_parser('conv_output',
                                   help='visualize output (feature maps) of convolution layers')
    parser.add_argument('-m', '--model-file', type=str, required=True,
                        help='keras model in HDF5 format')
    parser.add_argument('-l', '--layer-name', type=str, required=True,
                        help='name of the layer to visualize')
    parser.add_argument('-i', '--input-file', type=str, required=True,
                        help='input data file in HDF5 format')
    parser.add_argument('-n', '--n-samples', type=int, default=3,
                        help='number of samples to visualize')
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help='dataset name in the input HDF5 file')
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                        help='output directory to store images (<output_dir>/<sample_id>/<filter_id>.png)')
    args = main_parser.parse_args()

    if args.command == 'conv_output':
        from keras.models import load_model, Model
        import numpy as np
        from scipy.misc import imsave
        import h5py

        logger.info('load keras model from file: ' + args.model_file)
        model = load_model(args.model_file)
        logger.info('load input file: ' + args.input_file)
        fin = h5py.File(args.input_file, 'r')
        X = fin[args.dataset][:]
        fin.close()
        n_samples = min(args.n_samples, X.shape[0])
        X = np.take(X, np.arange(n_samples), axis=0)
        X = np.repeat(X, 3, axis=3)

        layer = model.get_layer(args.layer_name)
        if len(layer.output.shape) != 4:
            raise ValueError('expect rank =4 for output of layer %s, got %d'%(args.layer_name, len(layer.output.shape)))
        vis_model = Model(inputs=[model.input], outputs=[layer.output])
        y = vis_model.predict(X[:n_samples])
        logger.info('output shape of layer %s is %s'%(args.layer_name, repr(y.shape)))
        logger.info('save images to output directory: ' + args.output_dir)
        for i_sample in range(n_samples):
            for i_filter in range(y.shape[3]):
                output_file = os.path.join(args.output_dir, str(i_sample), '%04d.png'%i_filter)
                prepare_output_file(output_file)
                imsave(output_file, y[i_sample, :, :, i_filter])