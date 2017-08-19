#! C:\Python27\python.exe
#coding:utf-8
import argparse, sys, os, errno
import logging
from PIL import Image
import re

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('tojpg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract image and metadata from DICOM files')
    parser.add_argument('input_files', metavar='FILE', nargs='*',
                        help='input DICOM files')
    parser.add_argument('-i', '--input-dir', type=str,
                        help='input directory containing DICOM files')
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                        help='output directory')
    parser.add_argument('--suffix', type=str, default='.jpg',
                        help='suffix of DICOM files')
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    count=0
    input_files = []
    if input_files is not None:
        input_files += args.input_files
    if args.input_dir is not None:
        pathDir = os.listdir(args.input_dir)
        for allDir in pathDir:
            input=args.input_dir+'\\'+allDir
            count+=1
            logger.info('read png file: ' + input)
            file_id = os.path.splitext(os.path.basename(input))[0]
            img = Image.open(input)
            ims=img
            # w, h = ims.size
            #
            #
            #
            # if h<624 :
            #     w_tobox=(624/h*w)
            #     ims=ims.resize((int(w_tobox),624))
            # w, h = ims.size
            #
            # if w>512:
            #     box=(int(w/2-512/2),0,int(w/2+512/2),h)
            #     ims = ims.crop(box)
            #
            # w,h =ims.size
            # image1=ims
            # if w<512:
            #     image1 = Image.new("RGB", (512, 624))
            #     image1.paste(ims, ((512 - w)/2,0))

            pattern7 = re.compile(r'\d{7}')
            patternz = re.compile(ur'z|Z|正')
            patternc = re.compile(ur'c|C|侧')
            match7 = pattern7.search(allDir.decode("gbk"))
            matchz = patternz.search(allDir.decode("gbk"))
            matchc = patternc.search(allDir.decode("gbk"))
            output_file = os.path.join(args.output_dir, '%s' % (allDir))
            try:
                if matchz:
                    output_file = os.path.join(args.output_dir, '%s_Z.jpg' % (match7.group(0)))
                if matchc:
                    output_file = os.path.join(args.output_dir, '%s_C.jpg' % (match7.group(0)))
            except AttributeError:
                output_file = os.path.join(args.output_dir, '%s' % (allDir))

            # output_file = os.path.join(args.output_dir, '%s' % (allDir))
            ims.save(output_file)
            logger.info('save image: ' + output_file+'  count='+str(count))


        # plt.imshow(images[i],cmap='gray')
        # plt.axis('off')
        # plt.savefig(output_file,dpi=80,)

            #imsave(output_file, images[i])
