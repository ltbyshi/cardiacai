#! /usr/bin/env python
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s [%(levelname)s] : %(message)s')
logger = logging.getLogger('preprocess')

def prepare_output_file(filename):
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e

def resize_proportional(img, resize_width, resize_height):
    height, width = img.shape
    if width / height == resize_width / resize_height:
        img_new = cv2.resize(img, (resize_width, resize_height))
    elif width / height > resize_width / resize_height:
        img_new = np.zeros((resize_height, resize_width), dtype=img.dtype)
        height_new = resize_width * height / width
        img_resize = cv2.resize(img, (resize_width, height_new), interpolation=cv2.INTER_CUBIC)
        y0 = (resize_height - height_new) / 2
        img_new[y0:(y0 + height_new), :] = img_resize
    elif width / height < resize_width / resize_height:
        img_new = np.zeros((resize_height, resize_width), dtype=img.dtype)
        width_new = resize_height * width / height
        img_resize = cv2.resize(img, (width_new, resize_height), interpolation=cv2.INTER_CUBIC)
        x0 = (resize_width - width_new) / 2
        img_new[:, x0:(x0 + width_new)] = img_resize
    return img_new

def read_hdf5(filename, datasets):
    import h5py
    f = h5py.File(filename, 'r')
    if isinstance(datasets, list) or isinstance(datasets, tuple):
        data = []
        for dataset in datasets:
            data.append(f[dataset][:])
    else:
        data = f[datasets][:]
    f.close()
    return data

def image_to_hdf5(args):
    import cv2
    import numpy as np
    import h5py
    logger.info('create output file: ' + args.output_file)
    prepare_output_file(args.output_file)
    fout = h5py.File(args.output_file, 'w')
    if args.resize:
        resize_width, resize_height = [int(a) for a in args.resize.split(',')]
    images = []
    image_ids = []
    for input_file in args.input_files:
        logger.info('read input file: ' + input_file)
        img = cv2.imread(input_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if args.resize:
            img = resize_proportional(img, resize_width, resize_height)
        images.append(img.reshape((1, img.shape[0], img.shape[1], 1)))
        image_id = os.path.splitext(os.path.basename(input_file))[0]
        image_ids.append(image_id)
    images = np.vstack(images).astype('float32')
    images /= 255.0
    fout.create_dataset('X', data=images)
    fout.create_dataset('image_id', data=np.asarray(image_ids))
    fout.close()

def merge_image_hdf5(args):
    import numpy as np
    import h5py

    images = []
    image_ids = []
    for input_file in args.input_files:
        logger.info('read input file: ' + input_file)
        fin = h5py.File(input_file, 'r')
        images.append(fin['X'][:])
        image_ids.append(fin['image_id'][:])
    images = np.vstack(images)
    image_ids = np.concatenate(image_ids)
    logger.info('create output file: ' + args.output_file)
    prepare_output_file(args.output_file)
    fout = h5py.File(args.output_file, 'w')
    fout.create_dataset('X', data=images)
    fout.create_dataset('image_id', data=image_ids)
    fout.close()

def hdf5_to_image(args):
    import h5py
    import cv2
    import numpy as np
    from scipy.misc import imsave

    logger.info('read input file: ' + args.input_file)
    fin = h5py.File(args.input_file, 'r')
    images = fin[args.dataset][:]
    fin.close()

    if args.resize is not None:
        width, height = [int(a) for a in args.resize.split(',')]
    else:
        height, width = images.shape[1:3]
    ncol = args.ncol
    nrow = min(args.nrow, images.shape[0] / ncol)
    combined = np.full(((height + 2*args.margin) * nrow, (width + 2*args.margin)* ncol, images.shape[3]),
                       args.margin_color, dtype=images.dtype)
    for k in range(min(images.shape[0], nrow * ncol)):
        i = k / ncol
        j = k % ncol
        if args.resize is not None:
            image = cv2.resize(images[k], (width, height), interpolation=cv2.INTER_CUBIC)
        else:
            image = images[k]
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        y = i*(height + 2*args.margin) + args.margin
        x = j*(width + 2*args.margin) + args.margin
        combined[y:(y + height), x:(x + width)] = image
    logger.info('save combined image to file: ' + args.output_file)
    prepare_output_file(args.output_file)
    if combined.shape[-1] == 1:
        combined = np.squeeze(combined, axis=2)
    imsave(args.output_file, combined)

def download_model(args):
    if args.model_name == 'ResNet50':
        from keras.applications.resnet50 import ResNet50 as get_pretrained_model
    elif args.model_name == 'Xception':
        from keras.applications.xception import Xception as get_pretrained_model
    elif args.model_name == 'VGG16':
        from keras.applications.vgg16 import VGG16 as get_pretrained_model
    elif args.model_name == 'VGG19':
        from keras.applications.vgg19 import VGG19 as get_pretrained_model
    elif args.model_name == 'InceptionV3':
        from keras.applications.inception_v3 import InceptionV3 as get_pretrained_model
    logger.info('download pretrained model %s on ImageNet %s top layers' % (args.model_name,
                                                                            'with' if args.include_top else 'without'))
    input_shape = [int(a) for a in args.input_shape.split(',')]
    model = get_pretrained_model(include_top=args.include_top,
                                 weights='imagenet',
                                 input_shape=input_shape)
    logger.info('save model: ' + args.output_file)
    prepare_output_file(args.output_file)
    model.save(args.output_file)

def augment_images(args):
    import h5py
    import numpy as np
    from scipy.misc import imsave
    from keras.preprocessing.image import ImageDataGenerator

    logger.info('read input file: ' + args.input_file)
    fin = h5py.File(args.input_file, 'r')
    if args.dataset is not None:
        images = fin[args.dataset][:]
    elif len(fin.keys()) == 1:
        images = fin[fin.keys()[0]][:]
    else:
        raise ValueError('argument -d/--dataset is required if more than one dataset exist in the input file')
    fin.close()
    images = np.take(images, np.arange(args.n_samples), axis=0)
    labels = np.arange(images.shape[0])
    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=3,
        width_shift_range=0.1,
        height_shift_range=0.1,
        channel_shift_range=0.2,
        zoom_range=0.05,
        horizontal_flip=False)
    datagen.fit(images)
    X_aug = []
    y_aug = []
    logger.info('generate augmented images')
    i_batch = 0
    for X, y in datagen.flow(images, labels, batch_size=args.n_samples):
        X_aug.append(X)
        y_aug.append(y)
        i_batch += 1
        if i_batch >= args.n_images:
            break
    X_aug = np.vstack(X_aug)
    y_aug = np.concatenate(y_aug)

    logger.info('save images to file: ' + args.output_dir)
    """
    prepare_output_file(args.output_file)
    fout = h5py.File(args.output_file, 'w')
    fout.create_dataset('X', data=X_aug)
    fout.create_dataset('id', data=y_aug)
    fout.close()
    """
    for i_sample in range(args.n_samples):
        sample_directory = os.path.join(args.output_dir, str(i_sample))
        if not os.path.exists(sample_directory):
            os.makedirs(sample_directory)
        for i_batch, ind in enumerate(np.nonzero(y_aug == i_sample)[0]):
            imsave(os.path.join(sample_directory, '%d.png' % i_batch), np.squeeze(X_aug[ind]))

def create_dataset(args):
    import pandas as pd
    import h5py
    import numpy as np
    from sklearn.model_selection import train_test_split

    logger.info('read sample information from file: ' + args.input_file)
    sample_info = pd.read_excel(args.input_file)
    if args.task == 'classify_diseases':
        classes = args.classes.split(',')
        logger.info('defined %d classes: %s' % (len(classes), ','.join(classes)))
        query_str = '(position == "Z") and '
        filters = {}
        for c in classes:
            if c == 'normal_anzhen':
                filters[c] = '((diagnosis == "normal") and (data_source == "anzhen"))'
            elif c == 'normal_indiana':
                filters[c] = '((diagnosis == "normal") and (data_source == "indiana"))'
            else:
                filters[c] = '(diagnosis == "%s")' % c
        query_str = query_str + ' or '.join(filters.values())
        sample_info = sample_info.query(query_str)
        # one-hot coding for multiclass classification
        # one-dimensional output for two-class classification
        if len(classes) > 2:
            y = np.zeros((sample_info.shape[0], len(classes)), dtype='int32')
            for i, c in enumerate(classes):
                ind = (sample_info.eval(filters[c])).values
                y[ind, i] = 1
                logger.info('number of samples for Class %d (%s): %d' % (i, c, ind.sum()))
        else:
            y = np.zeros(sample_info.shape[0], dtype='int32')
            for i, c in enumerate(classes):
                ind = (sample_info.eval(filters[c])).values
                y[ind] = i
                logger.info('number of samples for Class %d (%s): %d' % (i, c, ind.sum()))
        image_id = sample_info['image_id'].values.astype('S')
        logger.info('create output file: ' + args.output_file)
        prepare_output_file(args.output_file)
        fout = h5py.File(args.output_file, 'w')
        fout.create_dataset('image_id', data=image_id)
        fout.create_dataset('y', data=y)
        fout.create_dataset('class_name', data=np.asarray(classes))
        fout.close()
    elif args.task == 'segment':
        sample_info = sample_info.query('(position == "Z") and (has_heart_trace)')
        if args.data_source is not None:
            sample_info = sample_info.query('data_source == "%s"' % args.data_source)
        logger.info('create output file: ' + args.output_file)
        prepare_output_file(args.output_file)
        fout = h5py.File(args.output_file, 'w')
        fout.create_dataset('image_id', data=sample_info['image_id'].values.astype('S'))
        fout.close()

def trace_to_mask(args):
    import h5py
    import numpy as np
    import cv2

    def read_polygon(filename):
        points = []
        with open(filename, 'r') as f:
            for line in f:
                points.append([int(a) for a in line.strip().split(',')])
        return np.asarray(points)

    if args.resize is not None:
        resize_width, resize_height = [int(a) for a in args.resize.split(',')]

    X = []
    image_ids = []
    for input_file in os.listdir(args.input_dir):
        if input_file.endswith(args.trace_suffix):
            logger.info('draw polygon from file: ' + os.path.join(args.input_dir, input_file))
            image_id = input_file.split('.')[0]
            polygon = read_polygon(os.path.join(args.input_dir, input_file))
            image = cv2.imread(os.path.join(args.image_dir, image_id + args.image_suffix))
            if args.mix:
                mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                mask = np.zeros(image.shape[:2], dtype='uint8')
            mask = cv2.fillPoly(mask, [polygon.reshape((-1, 1, 2))], color=[255, 255, 255])
            if args.resize:
                mask = resize_proportional(mask, resize_width, resize_height)
            mask = mask.reshape((1, mask.shape[0], mask.shape[1], 1))
            X.append(mask)
            image_ids.append(image_id)
    X = np.vstack(X)
    image_ids = np.asarray(image_ids)
    logger.info('create output file: ' + args.output_file)
    prepare_output_file(args.output_file)
    fout = h5py.File(args.output_file, 'w')
    fout.create_dataset('X', data=X)
    fout.create_dataset('image_id', data=image_ids)
    fout.close()

def cv_split(args):
    from sklearn.model_selection import KFold, train_test_split
    import numpy as np
    import h5py

    logger.info('read image ids from file: ' + args.input_file)
    image_ids = read_hdf5(args.input_file, 'image_id')
    if args.seed is not None:
        np.random.seed(args.seed)
    logger.info('create output file: ' + args.output_file)
    prepare_output_file(args.output_file)
    fout = h5py.File(args.output_file, 'w')
    i = 0
    kfold = KFold(args.k, shuffle=True)
    for indices_train, indices_test in kfold.split(image_ids):
        image_id_train = image_ids[indices_train]
        image_id_test = image_ids[indices_test]
        g = fout.create_group(str(i))
        if args.valid_size > 0:
            image_id_train, image_id_valid = train_test_split(image_id_train, test_size=args.valid_size)
            g.create_dataset('train', data=image_id_train)
            g.create_dataset('valid', data=image_id_valid)
        else:
            g.create_dataset('train', data=image_id_train)
        g.create_dataset('test', data=image_id_test)
        i += 1
    fout.close()

if __name__ == '__main__':
    main_parser = argparse.ArgumentParser(description='Infer parent genotypes from genotypes of descents')
    subparsers = main_parser.add_subparsers(dest='command')
    # command: image_to_hdf5
    parser = subparsers.add_parser('image_to_hdf5',
                                   help='convert images to gray-scale images and merge all images into one HDF5 file')
    parser.add_argument('-i', '--input-files', type=str, required=True, nargs='+',
                        help='input image files')
    parser.add_argument('-o', '--output-file', type=str, required=True,
                        help='output file')
    parser.add_argument('--resize', type=str,
                        help='comma-separated pair of integers (width, height). Resize the image.')
    # command: merge_image_hdf5
    parser = subparsers.add_parser('merge_image_hdf5')
    parser.add_argument('-i', '--input-files', type=str, nargs='+', required=True,
                        help='output files of image_to_hdf5')
    parser.add_argument('-o', '--output-file', type=str, required=True,
                        help='merged HDF5 file')
    # command: hdf5_to_image
    parser = subparsers.add_parser('hdf5_to_image',
                                   help='convert images in a HDF5 file to seperate or combined images')
    parser.add_argument('-i', '--input-file', type=str, required=True,
                        help='input HDF5 file')
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help='dataset name in the input HDF5 file')
    parser.add_argument('-r', '--nrow', type=int, default=1,
                        help='number of rows to tile')
    parser.add_argument('-c', '--ncol', type=int, default=1,
                        help='number of columns to tile')
    parser.add_argument('--resize', type=str,
                        help='comma-separated pair of integers (width, height). Resize the images.')
    parser.add_argument('--margin', type=int, default=0,
                        help='add marge around each image')
    parser.add_argument('--margin-color', type=float, default=0)
    parser.add_argument('-o', '--output-file', type=str, required=True,
                        help='output file')
    # command: histogram
    parser = subparsers.add_parser('image_histogram',
                                   help='get histogram of images')
    parser.add_argument('-i', '--input-files', type=str, required=True, nargs='+',
                        help='input image files')
    parser.add_argument('-t', '--type', type=str, required=True, nargs='+',
                        choices=('horizontal', 'vertical'))
    parser.add_argument('-o', '--output-file', type=str, required=True,
                        help='output histograms in HDF5 format')
    # command: download_model
    parser = subparsers.add_parser('download_model',
                                   help='download and save models provided in keras.applications')
    parser.add_argument('--model-name', type=str,
                        choices=['ResNet50', 'VGG19', 'VGG16', 'InceptionV3', 'Xception'])
    parser.add_argument('--include-top', action='store_true')
    parser.add_argument('--input-shape', type=str, default='320,320,3',
                        help='comma-separated of integers (rank = 3)')
    parser.add_argument('-o', '--output-file', type=str)
    # command: augment_images
    parser = subparsers.add_parser('augment_images',
                                   help='augment images by random transformations')
    parser.add_argument('-i', '--input-file', type=str, required=True,
                        help='input HDF5 file containing gray-scale images')
    parser.add_argument('-d', '--dataset', type=str,
                        help='dataset name in the input HDF5 file')
    parser.add_argument('-n', '--n-samples', type=int, default=10,
                        help='number of images to augment')
    parser.add_argument('--n-images', type=int, default=25,
                        help='number of images to augment')
    parser.add_argument('-o', '--output-dir', type=str)
    # command: create_dataset
    parser = subparsers.add_parser('create_dataset',
                                   help='get indices of training and test samples')
    parser.add_argument('-i', '--input-file', type=str, required=True,
                        help='an Excel table containing sample information')
    parser.add_argument('-t', '--task', type=str, required=True,
                        choices=('classify_diseases','segment'))
    parser.add_argument('--data-source', type=str, help='only keep samples from the data source')
    parser.add_argument('--has-heart-trace', action='store_true',
                        help='only use images with heart traces')
    parser.add_argument('-c', '--classes', type=str, default='normal,ASD',
                        help='comma-separated list of classes to classify')
    parser.add_argument('--test-size', type=int, required=False, default=0.1)
    parser.add_argument('-o', '--output-file', type=str, required=True,
                        help='output an HDF5 file containing training and test indices and target values.'
                        'Two datasets: image_id_train, image_id_test, y')
    # command: trace_to_mask
    parser = subparsers.add_parser('trace_to_mask',
                                   help='draw filled polygons from polygon paths')
    parser.add_argument('-i', '--input-dir', type=str, required=True,
                        help='input directory containing text files of polygon coordinates')
    parser.add_argument('--image-dir', type=str, required=True,
                        help='directory containing images with the same file name prefix with trace files')
    parser.add_argument('--trace-suffix', type=str, default='.jpg.txt',
                        help='suffix filter for file names')
    parser.add_argument('--image-suffix', type=str, default='.jpg')
    parser.add_argument('--mix', action='store_true',
                        help='use image as background for the mask')
    parser.add_argument('--resize', type=str,
                        help='comma-separated pair of integers (width, height). Resize the image.')
    parser.add_argument('-o', '--output-file', type=str, required=True,
                        help='output an HDF5 file containing training and test indices and target values.')
    # command: cv_split
    parser = subparsers.add_parser('cv_split',
                                   help='split a dataset into training/test datasets for k-fold cross-validation')
    parser.add_argument('-i', '--input-file', type=str, required=True,
                        help='an HDF5 file containing all image ids (dataset name: image_id)')
    parser.add_argument('-k', type=int, default=10,
                        help='number of folds for k-fold cross-validation')
    parser.add_argument('--valid-size', type=float, default=0.0,
                        help='fraction of training data for validation')
    parser.add_argument('--seed', type=int,
                        help='set seed for the random number generator')
    parser.add_argument('-o', '--output-file', type=str, required=True,
                        help='output an HDF5 file containing training and test ids (/<fold>/train, /<fold/test).')

    args = main_parser.parse_args()

    logger = logging.getLogger('preprocess.' + args.command)

    command_handlers = {
        'image_to_hdf5': image_to_hdf5,
        'hdf5_to_image': hdf5_to_image,
        'merge_image_hdf5': merge_image_hdf5,
        'create_dataset': create_dataset,
        'cv_split': cv_split,
        'trace_to_mask': trace_to_mask,
        'augment_images': augment_images,
        'download_model': download_model
    }
    import numpy as np
    import h5py

    command_handlers[args.command](args)

