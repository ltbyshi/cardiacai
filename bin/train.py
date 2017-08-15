#! /usr/bin/env python
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s [%(levelname)s] : %(message)s')

def prepare_output_file(filename):
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e

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


def array_lookup(mapping, keys):
    """
    Get indices of matched strings in a numpy array
    :param mapping: a 1D array of strings
    :param keys: a 1D array of keys to lookup
    :return: a 1D array of indices of keys
    """
    d = {k: i for i, k in enumerate(mapping)}
    return np.asarray([d[k] for k in keys])

from keras.preprocessing.image import apply_transform, transform_matrix_offset_center, random_channel_shift
from keras.preprocessing.image import ImageDataGenerator
class ImagePairDataGenerator(ImageDataGenerator):
    def random_transform_pair(self, x, y, seed=None):
        """Randomly augment a single image tensor.

                # Arguments
                    x: 3D tensor, single image.
                    seed: random seed.

                # Returns
                    A randomly transformed version of the input (same shape).
                """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        if seed is not None:
            np.random.seed(seed)

        # use composition of homographies
        # to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0

        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_axis]
        else:
            ty = 0

        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                     [0, np.cos(shear), 0],
                                     [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            h, w = x.shape[img_row_axis], x.shape[img_col_axis]
            transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
            x = apply_transform(x, transform_matrix, img_channel_axis,
                                fill_mode=self.fill_mode, cval=self.cval)
            y = apply_transform(y, transform_matrix, img_channel_axis,
                                fill_mode=self.fill_mode, cval=self.cval)
        if self.channel_shift_range != 0:
            x = random_channel_shift(x,
                                     self.channel_shift_range,
                                     img_channel_axis)

        return x, y

class BatchImageDataGenerator(object):
    def __init__(self, X, y, image_generator, batch_size=20, transform_y=False):
        self.batch_size = batch_size
        self.n_batches = int(np.ceil(float(y.shape[0])/batch_size))
        self.steps_per_epoch = self.n_batches
        self.transform_y = transform_y
        self.X = X
        self.y = y
        self.image_generator = image_generator
        self.n_samples = X.shape[0]
    def __call__(self):
        while True:
            indices = np.random.permutation(self.n_samples)
            for i in range(self.n_batches):
                indices_batch = indices[(i*self.batch_size):min((i + 1)*self.batch_size, self.n_samples)]
                X_batch = np.empty([len(indices_batch)] + list(self.X.shape[1:]), dtype=self.X.dtype)
                if self.transform_y:
                    y_batch = np.empty([len(indices_batch)] + list(self.y.shape[1:]), dtype=self.y.dtype)
                    for j, k in enumerate(indices_batch):
                        X_batch[i], y_batch[j] = self.image_generator.random_transform_pair(self.X[k], self.y[k])
                else:
                    for j, k in enumerate(indices_batch):
                        X_batch[j] = self.image_generator.random_transform(self.X[k])
                    y_batch = self.y[indices_batch]
                yield X_batch, y_batch

class UpsampleImageDataGenerator(object):
    def __init__(self, X, y, image_generator, batch_size=20):
        self.batch_size = 20
        n_positives = np.count_nonzero(y == 1)
        n_negatives = np.count_nonzero(y == 0)
        logger.info('number of positive/negative samples: %d/%d' % (n_positives, n_negatives))
        major_class = 1 if (n_positives > n_negatives) else 0
        self.indices_major = np.nonzero(y == major_class)[0]
        self.indices_minor = np.nonzero(y != major_class)[0]
        self.n_batches = np.round(len(self.indices_major) * 2 / batch_size)
        self.steps_per_epoch = self.n_batches
        self.X = X
        self.y = y
        self.image_generator = image_generator

    def __call__(self):
        while True:
            indices_major_rand = np.random.permutation(self.indices_major)
            indices_minor_rand = np.random.choice(self.indices_minor, replace=True, size=len(indices_major_rand))
            for i in range(self.n_batches):
                start = i * (self.batch_size / 2)
                end = (i + 1) * (self.batch_size / 2)
                indices_batch = np.concatenate([indices_major_rand[start:end],
                                                indices_minor_rand[start:end]])
                X_batch = np.empty([len(indices_batch)] + list(self.X.shape[1:]), dtype=self.X.dtype)
                for j, k in enumerate(indices_batch):
                    X_batch[j] = self.image_generator.random_transform(self.X[k])
                y_batch = self.y[indices_batch]
                yield X_batch, y_batch
def classify_types(args):
    from models import get_pretrained_vgg16
    import numpy as np
    import h5py
    import keras
    globals().update(locals())

    def data_generator(X, y, batch_size=25):
        n_samples = X.shape[0]
        while True:
            indices = np.random.permutation(n_samples)
            for i_batch in range(n_samples / batch_size):
                indices_batch = indices[(i_batch * batch_size):((i_batch + 1) * batch_size)]
                yield X[indices_batch], y[indices_batch]

    logger.info('read input file: ' + args.input_file)
    fin = h5py.File(args.input_file, 'r')
    X = fin['X'][:]
    positions = fin['position'][:]
    fin.close()

    y = (positions == 'C').astype('int32')
    logger.info('labels: %s' % str(positions[:10]))

    if args.indices_file is not None:
        fin = h5py.File(args.indices_file, 'r')
        indices = fin[args.indices_name][:]
        fin.close()
        X = np.take(X, indices, axis=0)
        y = np.take(y, indices, axis=0)
        logger.info('number of training samples: %d' % indices.shape[0])
    X = np.repeat(X, 3, axis=3)
    n_samples = X.shape[0]
    generator = data_generator(X, y, args.batch_size)

    logger.info('build model')
    input_shape = X.shape[1:]
    logger.info('input_shape = %s' % repr(X.shape))
    # model = get_model(args.model_name, input_shape)
    logger.info('load pretrained vgg16 model: data/VGG_imagenet.npy')
    model = get_pretrained_vgg16('data/VGG_imagenet.npy', input_shape)
    model.summary()
    logger.info('train the model')
    model.fit_generator(generator, steps_per_epoch=n_samples / args.batch_size,
                        epochs=args.epochs,
                        callbacks=[keras.callbacks.TensorBoard(log_dir=args.output_file + '.tensorboard')])
    logger.info('save the model: ' + args.output_file)
    prepare_output_file(args.output_file)
    model.save(args.output_file)

def classify_diseases(args):
    import keras
    import numpy as np
    from keras.preprocessing.image import ImageDataGenerator
    from keras.models import load_model
    from sklearn.metrics import accuracy_score, roc_auc_score
    from models import add_fc_layers
    import h5py
    globals().update(locals())

    logger.info('read cv_split file: ' + args.cv_split_file)
    logger.info('use cv fold: %d' % args.cv_fold)
    fin = h5py.File(args.cv_split_file, 'r')
    image_id_train = fin['/%d/train' % args.cv_fold][:]
    image_id_test = fin['/%d/test' % args.cv_fold][:]
    image_id_valid = fin['/%d/valid' % args.cv_fold][:]
    fin.close()

    logger.info('read input images file: ' + args.input_file)
    fin = h5py.File(args.input_file, 'r')
    X = fin['X'][:]
    image_id_X = fin['image_id'][:]
    fin.close()
    logger.info('convert gray-scale images to 3-channel images')
    X_train = np.repeat(X[array_lookup(image_id_X, image_id_train)], 3, axis=3)
    X_test = np.repeat(X[array_lookup(image_id_X, image_id_test)], 3, axis=3)
    X_valid = np.repeat(X[array_lookup(image_id_X, image_id_valid)], 3, axis=3)
    del X

    logger.info('read targets file: ' + args.target_file)
    fin = h5py.File(args.target_file, 'r')
    y = fin['y'][:]
    image_id_y = fin['image_id'][:]
    fin.close()
    y_train = np.take(y, array_lookup(image_id_y, image_id_train), axis=0)
    y_test = np.take(y, array_lookup(image_id_y, image_id_test), axis=0)
    y_valid = np.take(y, array_lookup(image_id_y, image_id_valid), axis=0)
    del y

    if args.mask_file is not None:
        logger.info('read mask data from file: ' + args.mask_file)
        fin = h5py.File(args.mask_file, 'r')
        mask = fin['X'][:]
        image_id_mask = fin['image_id'][:]
        fin.close()
        # set
        mask = mask.astype('float32')
        mask = np.clip(mask, 0.2, 1.0)
        logger.info('apply mask to input images')
        X_train *= mask[array_lookup(image_id_mask, image_id_train)]
        X_test *= mask[array_lookup(image_id_mask, image_id_test)]
        X_valid *= mask[array_lookup(image_id_mask, image_id_valid)]
        del mask

    # multi-class
    if len(y_train.shape) > 1:
        n_classes = y_train.shape[1]
        class_freq = {c: count for c, count in enumerate(y_train.sum(axis=0))}
    # two-class
    else:
        class_freq = {c: count for c, count in zip(*np.unique(y_train, return_counts=True))}
        n_classes = 2
    logger.info('number of classes: %d' % (n_classes))
    logger.info('class frequencies in training data: ' + repr(class_freq))

    image_generator = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=3,
        width_shift_range=0.1,
        height_shift_range=0.1,
        channel_shift_range=0.2,
        zoom_range=0.1,
        horizontal_flip=False)

    logger.info('create batch data generator')
    data_generator = BatchImageDataGenerator(X_train, y_train,
                                             image_generator=image_generator,
                                             batch_size=args.batch_size,
                                             transform_y=False)

    logger.info('read model file: ' + args.pretrained_model_file)
    pretrained_model = load_model(args.pretrained_model_file)
    if args.fine_tune:
        logger.info('fix weights in pretrained model for fine-tuning')
        for layer in pretrained_model.layers:
            layer.trainable = False
    logger.info('add FC layers for classification')
    model = add_fc_layers(pretrained_model, n_classes)
    model.summary()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    train_log_file = os.path.join(args.output_dir, 'train.log')
    logger.info('train the model')
    callbacks = [keras.callbacks.CSVLogger(train_log_file)]

    class_weight = {c: 1.0 / float(class_freq[c]) for c in class_freq.keys()}
    class_weight_norm = sum(class_weight.values())
    class_weight = {c: class_weight[c] / class_weight_norm for c in class_weight.keys()}
    logger.info('class weight: ' + repr(class_weight))
    """
    datagen = data_generator()
    for i in range(50):
        X_batch, y_batch = datagen.next()
        model.train_on_batch(X_batch, y_batch, class_weight=class_weight)
        logger.info('evaluate on batch %d (size: %d): %s'%(i, X_batch.shape[0], repr(model.evaluate(X_batch, y_batch))))
    """
    model.fit_generator(data_generator(),
                        steps_per_epoch=data_generator.steps_per_epoch,
                        epochs=args.epochs,
                        callbacks=callbacks,
                        class_weight=class_weight,
                        validation_data=(X_valid, y_valid))
    logger.info('test the model')
    y_pred = model.predict(X_test, batch_size=args.batch_size)
    if n_classes > 2:
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_test, axis=1)
    else:
        y_pred_labels = (np.ravel(y_pred) > 0.5).astype('int32')
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_test, y_pred_labels)
    if n_classes > 2:
        metrics['accuracy_by_class'] = np.asarray([accuracy_score((y_test == c).astype('int32'),
                                                                  (y_pred_labels == c).astype('int32')) for c in
                                                   range(n_classes)])
    else:
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred)

    pred_file = os.path.join(args.output_dir, 'predictions')
    logger.info('save predictions to file: ' + pred_file)
    fout = h5py.File(pred_file, 'w')
    fout.create_dataset('y_true', data=y_test)
    fout.create_dataset('y_pred', data=y_pred)
    fout.create_dataset('y_pred_labels', data=y_pred_labels)
    g = fout.create_group('metrics')
    for key in metrics:
        logger.info('on test data, %s = %s' % (key, repr(metrics[key])))
        g.create_dataset(key, data=metrics[key])
    fout.close()

    model_file = os.path.join(args.output_dir, 'model')
    logger.info('save the model to file: ' + model_file)
    prepare_output_file(model_file)
    model.save(model_file)

def segment(args):
    from models import unet1
    import keras
    import numpy as np
    import h5py
    globals().update(locals())

    logger.info('read cv_split file: ' + args.cv_split_file)
    logger.info('use cv fold: %d' % args.cv_fold)
    fin = h5py.File(args.cv_split_file, 'r')
    image_id_train = fin['/%d/train' % args.cv_fold][:]
    image_id_test = fin['/%d/test' % args.cv_fold][:]
    image_id_valid = fin['/%d/valid' % args.cv_fold][:]
    fin.close()

    logger.info('read image data from file: ' + args.input_file)
    fin = h5py.File(args.input_file, 'r')
    X = fin['X'][:]
    image_id_X = fin['image_id'][:]
    fin.close()
    if args.model_name in ('unet1',):
        X_train = np.take(X, array_lookup(image_id_X, image_id_train), axis=0)
        X_test = np.take(X, array_lookup(image_id_X, image_id_test), axis=0)
        X_valid = np.take(X, array_lookup(image_id_X, image_id_valid), axis=0)
    else:
        logger.info('convert RGB images to gray-scale images')
        X_train = np.repeat(X[array_lookup(image_id_X, image_id_train)], 3, axis=3)
        X_test = np.repeat(X[array_lookup(image_id_X, image_id_test)], 3, axis=3)
        X_valid = np.repeat(X[array_lookup(image_id_X, image_id_valid)], 3, axis=3)
    del X

    logger.info('read mask data from file: ' + args.mask_file)
    fin = h5py.File(args.mask_file, 'r')
    y = fin['X'][:]
    image_id_y = fin['image_id'][:]
    fin.close()
    y = np.clip(y, 0, 1)
    y_train = np.take(y, array_lookup(image_id_y, image_id_train), axis=0)
    y_test = np.take(y, array_lookup(image_id_y, image_id_test), axis=0)
    y_valid = np.take(y, array_lookup(image_id_y, image_id_valid), axis=0)
    del y

    image_generator = ImagePairDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=3,
        width_shift_range=0.1,
        height_shift_range=0.1,
        channel_shift_range=0.2,
        zoom_range=0.1,
        horizontal_flip=False)
    data_generator = BatchImageDataGenerator(X_train, y_train,
                                             image_generator=image_generator,
                                             batch_size=args.batch_size,
                                             transform_y=False)

    logger.info('build model: unet1')
    model = unet1(input_shape=X_train.shape[1:])
    model.summary()
    logger.info('train the model')
    if not os.path.exists(args.output_dir):
        logger.info('create output directory: ' + args.output_dir)
        os.makedirs(args.output_dir)
    train_log_file = os.path.join(args.output_dir, 'train.log')
    callbacks = [keras.callbacks.CSVLogger(train_log_file)]
    model.fit_generator(data_generator(),
                        validation_data=(X_valid, y_valid),
                        steps_per_epoch=data_generator.steps_per_epoch,
                        epochs=args.epochs,
                        callbacks=callbacks)

    model_file = os.path.join(args.output_dir, 'model')
    logger.info('save the model to file: ' + model_file)
    model.save(model_file)

if __name__ == '__main__':
    main_parser = argparse.ArgumentParser(description='Train models for classification of chest X-ray radiography')
    subparsers = main_parser.add_subparsers(dest='command')
    # command: classify_types
    parser = subparsers.add_parser('classify_types',
                                   help='a simple classifier for types')
    parser.add_argument('-i', '--input-file', type=str, required=True,
                        help='input HDF5 file')
    parser.add_argument('--indices-file', type=str,
                        help='HDF5 file containing training indices')
    parser.add_argument('--indices-name', type=str,
                        help='HDF5 dataset name of training indices')
    parser.add_argument('--suffix', type=str, default='.jpg',
                        help='suffix of input file names')
    parser.add_argument('-m', '--model-name', type=str, choices=('vgg16',), default='vgg16',
                        help='model name')
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('-o', '--output-file', type=str, required=True,
                        help='output model file')

    # command: classify_diseases
    parser = subparsers.add_parser('classify_diseases')
    parser.add_argument('-i', '--input-file', type=str, required=True,
                        help='input HDF5 file')
    parser.add_argument('-y', '--target-file', type=str, required=True,
                        help='an HDF5 file containing target values')
    parser.add_argument('--mask-file', type=str,
                        help='an HDF5 file containing mask images of the same shape of input images')
    parser.add_argument('--cv-split-file', type=str,
                        help='HDF5 file containing training indices')
    parser.add_argument('--cv-fold', type=int, default=0)
    parser.add_argument('-m', '--pretrained-model-file', type=str,
                        help='pretrained keras model file')
    parser.add_argument('--fine-tune', action='store_true',
                        help='only tune the parameters of the fully-connected layers')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                        help='output directory for saving models and predictions')

    # command: segment
    parser = subparsers.add_parser('segment')
    parser.add_argument('-i', '--input-file', type=str, required=True,
                        help='input HDF5 file containing images to segment')
    parser.add_argument('-y', '--mask-file', type=str, required=True,
                        help='an HDF5 file containing mask images of the same shape of input images')
    parser.add_argument('--model-name', type=str, default='unet1',
                        choices=('unet1',),
                        help='the name of the model to use')
    parser.add_argument('--cv-split-file', type=str,
                        help='HDF5 file containing training indices')
    parser.add_argument('--cv-fold', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                        help='output directory for saving models and predictions')

    args = main_parser.parse_args()
    logger = logging.getLogger('train.' + args.command)

    command_handlers = {
        'classify_types': classify_types,
        'classify_diseases': classify_diseases,
        'segment': segment
    }
    import numpy as np
    import h5py
    command_handlers[args.command](args)