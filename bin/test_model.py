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

    parser = subparsers.add_parser('unet_vgg16',
                                   help='a simple classifier for types')
    parser.add_argument('-m', '--model-file', type=str, required=True)
    parser.add_argument('-o', '--output-file', type=str, required=True,
                        help='output model file')
    args = main_parser.parse_args()

    logger = logging.getLogger('test_model.' + args.command)

    if args.command == 'unet_vgg16':
        from models import unet_from_vgg16
        from keras.models import load_model
        from keras.utils.vis_utils import plot_model

        model = load_model(args.model_file)
        model = unet_from_vgg16(model)
        plot_model(model, args.output_file, show_shapes=True)




