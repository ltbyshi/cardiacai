#! /usr/bin/env python
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s [%(levelname)s] : %(message)s')
logger = logging.getLogger('report')

def summarize_cv(args):
    import pandas as pd
    columns = {}
    colnames = {}
    with h5py.File(os.path.join(args.input_dir, 'cv_split'), 'r') as f:
        n_folds = len(f.keys())
        cv_split = {}
        for cv_fold in f.keys():
            cv_split[int(cv_fold)] = {}
            g = f[cv_fold]
            for key in g.keys():
                cv_split[int(cv_fold)][key] = g[key][:]
    colnames['classification'] = []
    if args.task == 'classification':
        with h5py.File(os.path.join(args.input_dir, 'targets'), 'r') as f:
            class_name = f['class_name'][:]
            y = f['y'][:]
            image_id_y = f['image_id'][:]
        colnames['classification'] = ['train_size', 'test_size']
        colnames['classification'] += ['class_size(%s)'%(class_name[i]) for i in range(len(class_name))]
    columns['cv_fold'] = np.full(n_folds, -1, dtype='int32')
    colnames['metric'] = []
    for cv_fold in range(n_folds):
        cv_dir = os.path.join(args.input_dir, 'cv', str(cv_fold))
        columns['cv_fold'][cv_fold] = cv_fold
        if not os.path.isdir(cv_dir):
            continue
        pred_file = os.path.join(cv_dir, 'predictions')
        with h5py.File(pred_file, 'r') as f:
            g = f['metrics']
            # get column names
            if len(colnames['metric']) == 0 :
                colnames['metric'] = []
                for metric in g.keys():
                    if len(g[metric].shape) == 0:
                        colnames['metric'].append(metric)
                    elif metric == 'accuracy_by_class':
                        colnames['metric'] += ['%s(%s)'%(metric, class_name[i]) for i in range(g[metric].shape[0])]
                for metric in colnames['metric']:
                    columns[metric] = np.full(n_folds, np.nan, dtype='float64')
                if args.task == 'classification':
                    for colname in ['train_size', 'test_size']:
                        columns[colname] = np.zeros(n_folds, dtype='int32')
                    for i in range(len(class_name)):
                        columns['class_size(%s)'%(class_name[i])] = np.zeros(n_folds, dtype='int32')
            for metric in g.keys():
                if len(g[metric].shape) == 0:
                    columns[metric][cv_fold] = g[metric][()]
                elif metric == 'accuracy_by_class':
                    metric_vals = g[metric][:]
                    for i in range(g[metric].shape[0]):
                        columns['%s(%s)'%(metric, class_name[i])][cv_fold] = metric_vals[i]
            if args.task == 'classification':
                columns['train_size'][cv_fold] = cv_split[cv_fold]['train'].shape[0]
                columns['test_size'][cv_fold] = cv_split[cv_fold]['test'].shape[0]
                for i in range(len(class_name)):
                    y_test = y[array_lookup(image_id_y, cv_split[cv_fold]['test'])]
                    # one-hot coding for multi-class
                    if len(y_test.shape) > 1:
                        columns['class_size(%s)'%(class_name[i])][cv_fold] = np.sum(y_test[:, i])
                    # two-class
                    else:
                        columns['class_size(%s)' % (class_name[i])][cv_fold] = np.sum(y_test == i)
    summary = pd.DataFrame(columns)
    attribute_keys = []
    if args.attribute is not None:
        for a in args.attribute:
            if '=' not in a:
                raise ValueError('missing = in attribute: ' + a)
            ind = a.index('=')
            key = a[:ind].strip()
            val = a[(ind + 1):].strip()
            summary[key] = val
            attribute_keys.append(key)
    summary = summary[attribute_keys + ['cv_fold'] + colnames['classification'] + colnames['metric']]

    logger.info('create output file: ' + args.output_file)
    prepare_output_file(args.output_file)
    summary.to_csv(args.output_file, sep='\t', quoting=False, index=False)

if __name__ == '__main__':
    main_parser = argparse.ArgumentParser(description='Generate reports')
    subparsers = main_parser.add_subparsers(dest='command')

    # command: summarize_cv
    parser = subparsers.add_parser('summarize_cv',
                                   help='summarize parameters, metrics for cross-validation')
    parser.add_argument('-i', '--input-dir', type=str, required=True,
                        help='cross-validation directory with directory structure like: cv/<cv_fold>/predictions')
    parser.add_argument('-a', '--attribute', type=str, action='append',
                        help='key=value pairs to add to the columns')
    parser.add_argument('-t', '--task', type=str, default='classification')
    parser.add_argument('-o', '--output-file', type=str, required=True,
                        help='an output text report file')

    args = main_parser.parse_args()

    logger = logging.getLogger('report.' + args.command)
    command_handlers = {
        'summarize_cv': summarize_cv
    }
    import numpy as np
    import h5py
    from utils import prepare_output_file, array_lookup
    command_handlers[args.command](args)

