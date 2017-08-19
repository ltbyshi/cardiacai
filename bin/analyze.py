#! /usr/bin/env python
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s [%(levelname)s] : %(message)s')
logger = logging.getLogger('analyze')


def plot_model(args):
    import keras

    logger.info('load model from file: ' + args.model_file)
    model = keras.models.load_model(args.model_file)
    logger.info('save plot model file: ' + args.output_file)
    prepare_output_file(args.output_file)
    keras.utils.plot_model(model, args.output_file, show_shapes=True)

def deep_taylor(model, X, y):
    import tensorflow as tf
    import lrp
    import keras.backend as K

    logger.info('input tensor name: ' + model.input.name.split(':')[0])
    logger.info('output tensor name: ' + model.output.name.split(':')[0])
    y_true = tf.placeholder(dtype=model.output.dtype, shape=model.output.shape, name='truth')
    # multi-class  prediction
    if model.output.shape[1] > 2:
        R = model.output * y_true
    # predict the positive class
    else:
        R = model.output
    F_list = lrp.lrp(R, 0, 1, return_flist=True,
                     input_tensor_name=model.input.name.split(':')[0],
                     output_tensor_name=model.output.name.split(':')[0])
    sess = K.get_session()
    relevance_maps = lrp.get_lrp_im(sess, F_list[-1], model.input, y_true, X, y)
    logger.info('normalize LRP images')
    X = np.expand_dims(X[:, :, :, 0], axis=3)
    # convert the LRP image to gray-scale and normalize to range [0, 1]
    for i in range(len(relevance_maps)):
        relevance_maps[i] = np.mean(relevance_maps[i], axis=2)
        relevance_maps[i] = np.clip(relevance_maps[i], 0, np.percentile(relevance_maps[i], 99))
        relevance_maps[i] = relevance_maps[i] / (relevance_maps[i].max() + 1)
    relevance_maps = np.concatenate(relevance_maps).reshape(X.shape)
    return relevance_maps

def sensitivity_analysis(model, X, y, batch_size=32):
    import tensorflow as tf
    import lrp
    import keras.backend as K

    y_true = tf.placeholder(dtype=model.output.dtype, shape=model.output.shape, name='truth')
    # multi-class  prediction
    if model.output.shape[1] > 2:
        R = model.output * y_true
    # predict the positive class
    else:
        R = model.output
    grad_to_X = tf.gradients(R, model.input)
    sess = K.get_session()
    n_batches = int(np.ceil(float(X.shape[0])/batch_size))
    grad_val = np.empty(X.shape, dtype=X.dtype)
    for i in range(n_batches):
        start = i*batch_size
        end = min((i + 1)*batch_size, X.shape[0])
        grad_val[start:end] = sess.run(grad_to_X,
                            feed_dict={model.input: X[start:end], y_true: y[start:end]})[0]
    # take the average over multiple channels
    grad_val **= 2
    if grad_val.shape[3] > 1:
        grad_val = np.expand_dims(grad_val.mean(axis=3), axis=3)
    # normalize the gradients to range [0, 1]

    grad_min = np.percentile(grad_val.reshape((grad_val.shape[0], -1)), 5, axis=1)
    grad_max = np.percentile(grad_val.reshape((grad_val.shape[0], -1)), 95, axis=1)
    for i in range(grad_val.shape[0]):
        grad_val[i] = np.clip(grad_val[i], grad_min[i], grad_max[i])
        grad_val[i] -= grad_min[i]
        grad_val[i] /= (grad_max[i] - grad_min[i])

    if False:
        grad_min = grad_val.reshape((grad_val.shape[0], -1)).min(axis=1).reshape((-1, 1, 1, 1))
        grad_max = grad_val.reshape((grad_val.shape[0], -1)).max(axis=1).reshape((-1, 1, 1, 1))
        grad_val -= grad_min
        grad_val /= (grad_max - grad_min)
    return grad_val

def analyze_relevance(args):
    import keras
    from utils import read_hdf5, array_lookup

    logger.info('load model from file: ' + args.model_file)
    model = keras.models.load_model(args.model_file)
    logger.info('read image id file: ' + args.image_id_file)
    image_id = read_hdf5(args.image_id_file, args.image_id_dataset)
    logger.info('read input file: ' + args.input_file)
    X, image_id_X = read_hdf5(args.input_file, [args.input_dataset, 'image_id'])
    if model.input.shape[3] > 1:
        logger.info('convert gray-scale images to 3-channel images')
        X = np.repeat(X[array_lookup(image_id_X, image_id)], 3, axis=3)
    else:
        X = np.take(X, array_lookup(image_id_X, image_id), axis=0)
    logger.info('read target file: ' + args.target_file)
    y, image_id_y = read_hdf5(args.target_file, [args.target_dataset, 'image_id'])
    y = np.take(y, array_lookup(image_id_y, image_id), axis=0)
    if len(y.shape) == 1:
        y = np.expand_dims(y, axis=1)


    if args.method == 'deep_taylor':
        logger.info('input tensor name: ' + model.input.name.split(':')[0])
        logger.info('output tensor name: ' + model.output.name.split(':')[0])
        logger.info('start Deep Taylor decomposition')
        relevance_maps = deep_taylor(model, X, y)
    elif args.method == 'sensitivity':
        logger.info('start sensitivity analysis')
        relevance_maps = sensitivity_analysis(model, X, y, args.batch_size)

    logger.info('save relevance maps to file: ' + args.output_file)
    prepare_output_file(args.output_file)
    fout = h5py.File(args.output_file, 'w')
    fout.create_dataset('X', data=X)
    fout.create_dataset('relevance_map', data=relevance_maps)
    fout.create_dataset('image_id', data=image_id)
    fout.close()

if __name__ == '__main__':
    main_parser = argparse.ArgumentParser(description='Analyze the results and diagnosis')
    subparsers = main_parser.add_subparsers(dest='command')

    # command: plot_model
    parser = subparsers.add_parser('plot_model',
                                   help='plot the structure of a saved keras model')
    parser.add_argument('-m', '--model-file', type=str, required=True,
                        help='a saved keras model file')
    parser.add_argument('-o', '--output-file', type=str, required=True,
                        help='output image file')

    # command:
    parser = subparsers.add_parser('analyze_relevance',
                                   help='Layerwise Relevance Propagation with Deep Taylor Series')
    parser.add_argument('-i', '--input-file', type=str, required=True,
                        help='input data')
    parser.add_argument('--input-dataset', type=str, required=True,
                        help='dataset name in the input file')
    parser.add_argument('-y', '--target-file', type=str, required=True,
                        help='an HDF5 file containing target values')
    parser.add_argument('--target-dataset', type=str, required=True,
                        help='dataset name in the target file')
    parser.add_argument('--image-id-file', type=str, required=True,
                        help='an HDF5 file containing image ids to analyze')
    parser.add_argument('--image-id-dataset', type=str, required=True,
                        help='dataset name in the image id file')
    parser.add_argument('-m', '--model-file', type=str, required=True,
                        help='a saved keras model file')
    parser.add_argument('-o', '--output-file', type=str, required=True,
                        help='output image file')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size for prediction')
    parser.add_argument('--method', type=str, default='deep_taylor',
                        choices=('deep_taylor', 'sensitivity'),
                        help='method to use for relevance analysis')

    args = main_parser.parse_args()

    logger = logging.getLogger('train.' + args.command)
    command_handlers = {
        'plot_model': plot_model,
        'analyze_relevance': analyze_relevance,
    }
    import numpy as np
    import h5py
    from utils import prepare_output_file
    command_handlers[args.command](args)