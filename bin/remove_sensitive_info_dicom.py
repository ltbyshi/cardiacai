#! /usr/bin/env python
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('remove_sensitive_info_dicom')
import dicom

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_files', metavar='FILE', nargs='*',
                        help='input DICOM files')
    parser.add_argument('-i', '--input-dir', type=str,
                        help='input directory containing DICOM files')
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                        help='output directory')
    parser.add_argument('--suffix', type=str, default='.img',
                        help='suffix of DICOM files')
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
        ds = dicom.read_file(filename)
        invalid_tags = []
        for tag in ds.keys():
            try:
                ds[tag]
            except ValueError:
                invalid_tags.append(tag)
        for tag in invalid_tags:
            del ds[tag]
        for name in ['PatientName', 'InstitutionName', 'InstitutionAddress']:
            try:
                del ds[ds.data_element(name).tag]
            except ValueError:
                logger.warn('cannot remove data element: ' + name)
                pass
        all_tags = ds.keys()
        for tag in all_tags:
            if ds[tag].description() in ('Private tag data',):
                del ds[tag]
        output_file = os.path.join(args.output_dir, os.path.basename(filename))
        logger.info('save file: ' + output_file)
        ds.save_as(output_file)