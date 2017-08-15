#! /usr/bin/env python
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('extract_dicom')
import dicom
import numpy as np
from scipy.misc import imsave

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract image and metadata from DICOM files')
    parser.add_argument('input_files', metavar='FILE', nargs='*',
                        help='input DICOM files')
    parser.add_argument('-i', '--input-dir', type=str,
                        help='input directory containing DICOM files')
    parser.add_argument('--suffix', type=str, default='.img',
                        help='suffix of DICOM files')
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                        help='output directory')
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    input_files = []
    if input_files is not None:
        input_files += args.input_files
    if args.input_dir is not None:
        for root, dirs, files in os.walk(args.input_dir):
            for filename in files:
                if filename.endswith(args.suffix):
                    input_files.append(os.path.join(root, filename))

    for filename in input_files:
        logger.info('read DICOM file: ' + filename)
        file_id = os.path.splitext(os.path.basename(filename))[0]
        ds = dicom.read_file(filename)
        images = ds.pixel_array
        if len(images.shape) < 3:
            images = np.expand_dims(images, axis=0)
        images[images == images.max()] = 0
        for i in range(images.shape[0]):
            output_file = os.path.join(args.output_dir, '%s.%s.%d.png'%(ds.PatientID, file_id, i))
            logger.info('save image: ' + output_file)
            imsave(output_file, images[i])
