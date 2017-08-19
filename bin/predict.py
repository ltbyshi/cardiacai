#! /usr/bin/env python
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s [%(levelname)s] : %(message)s')
logger = logging.getLogger('train')

def segment(args):
    from models import custom_objects
    from keras.models import load_model
    import numpy as np
    import h5py
    globals().update(locals())

    logger.info('load model from file: ' + args.model_file)
    model = load_model(args.model_file, custom_objects=custom_objects)

    logger.info('read image ids from file: ' + args.image_id_file)
    image_id = read_hdf5(args.image_id_file, args.image_id_dataset)

    logger.info('read image data from file: ' + args.input_file)
    fin = h5py.File(args.input_file, 'r')
    X = fin['X'][:]
    image_id_X = fin['image_id'][:]
    fin.close()
    if model.input.shape[3] > 1:
        logger.info('convert gray-scale images to 3-channel images')
        X = np.repeat(X[array_lookup(image_id_X, image_id)], 3, axis=3)
    else:
        X = np.take(X, array_lookup(image_id_X, image_id), axis=0)

    logger.info('predict')
    y = model.predict(X, batch_size=args.batch_size)
    if args.mix:
        logger.info('mix output masks with input images')
        y = np.clip(y + X, 0, 1)

    logger.info('create output file: ' + args.output_file)
    prepare_output_file(args.output_file)
    fout = h5py.File(args.output_file, 'w')
    fout.create_dataset('X', data=y)
    fout.create_dataset('image_id', data=image_id)
    fout.close()

def classify_diseases(args):
    from keras.models import load_model
    from models import custom_objects
    import numpy as np
    import h5py
    globals().update(locals())

    logger.info('load model from file: ' + args.model_file)
    model = load_model(args.model_file, custom_objects=custom_objects)

    logger.info('read image ids from file: ' + args.image_id_file)
    image_id = read_hdf5(args.image_id_file, args.image_id_dataset)

    logger.info('read image data from file: ' + args.input_file)
    fin = h5py.File(args.input_file, 'r')
    X = fin['X'][:]
    image_id_X = fin['image_id'][:]
    fin.close()
    if model.input.shape[3] > 1:
        logger.info('convert gray-scale images to 3-channel images')
        X = np.repeat(X[array_lookup(image_id_X, image_id)], 3, axis=3)
    else:
        X = np.take(X, array_lookup(image_id_X, image_id), axis=0)

    logger.info('predict')
    y = model.predict(X, batch_size=args.batch_size)

    logger.info('create output file: ' + args.output_file)
    prepare_output_file(args.output_file)
    fout = h5py.File(args.output_file, 'w')
    fout.create_dataset('y', data=y)
    fout.create_dataset('image_id', data=image_id)
    fout.close()

if __name__ == '__main__':
    main_parser = argparse.ArgumentParser(description='Predict using saved models')
    subparsers = main_parser.add_subparsers(dest='command')

    # command: classify_diseases
    parser = subparsers.add_parser('classify_diseases')

    # command: segment
    parser = subparsers.add_parser('segment')
    parser.add_argument('-i', '--input-file', type=str, required=True,
                        help='input HDF5 file containing images to segment')
    parser.add_argument('-m', '--model-file', type=str, required=True,
                        help='a saved keras model file')
    parser.add_argument('-y', '--mask-file', type=str,
                        help='an HDF5 file containing mask images of the same shape of input images')
    parser.add_argument('--image-id-file', type=str,
                        help='HDF5 file containing image ids')
    parser.add_argument('--image-id-dataset', type=str,
                        help='HDF5 dataset name in the file specified by --image-id-file')
    parser.add_argument('--mix', action='store_true',
                        help='mix the segmentation result with input')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('-o', '--output-file', type=str, required=True,
                        help='output directory for saving models and predictions')

    args = main_parser.parse_args()
    command_handlers = {
        'segment': segment
    }
    logger = logging.getLogger('predict.' + args.command)
    import numpy as np
    from utils import read_hdf5, prepare_output_file, array_lookup
    command_handlers[args.command](args)