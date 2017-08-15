#! /usr/bin/env python
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s [%(levelname)s] : %(message)s')
logger = logging.getLogger('analyze')

def prepare_output_file(filename):
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e

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
    args = main_parser.parse_args()

    if args.command == 'plot_model':
        from keras.models import load_model
        from keras.utils import plot_model

        logger.info('load model from file: ' + args.model_file)
        model = load_model(args.model_file)
        logger.info('save plot model file: ' + args.output_file)
        prepare_output_file(args.output_file)
        plot_model(model, args.output_file, show_shapes=True)