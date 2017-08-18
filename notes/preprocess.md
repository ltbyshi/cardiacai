## Get image sizes
```bash
bin/preprocess.py get_image_size -i data/anzhen/images/*.jpg | tee output/anzhen/image_size/raw.txt
```
## Convert images to gray-scale and convert to equal size (320x320)
```bash
# anzhen data
bin/preprocess.py image_to_hdf5 -i data/anzhen/process_512x624/*.jpg -o data/anzhen/datasets/abnormal_Z_320x320.h5 --resize 320,320
bin/preprocess.py hdf5_to_image -i data/anzhen/datasets/abnormal_Z_320x320.h5 -d X --nrow 10 --ncol 4 -o tmp/preview/anzhen_abnormal_Z_320x320.png
# anzhen data (normal)
bin/preprocess.py image_to_hdf5 -i data/anzhen/normal_Z/noline/*.jpg -o data/anzhen/datasets/normal_Z_320x320.h5 --resize 320,320
bin/preprocess.py hdf5_to_image -i data/anzhen/datasets/normal_Z_320x320.h5 -d X --nrow 10 --ncol 4 -o tmp/preview/anzhen_normal_Z_320x320.png
# indiana data (normal)
bin/preprocess.py image_to_hdf5 -i data/indiana/normal_Z/noline/*.jpg -o data/indiana/datasets/normal_Z_320x320.h5 --resize 320,320
bin/preprocess.py hdf5_to_image -i data/indiana/datasets/normal_Z_320x320.h5 -d X --nrow 10 --ncol 4 -o tmp/preview/indiana_normal_Z_320x320.png
# indiana data (abnormal)
bin/preprocess.py image_to_hdf5 -i data/indiana/abnormal_Z/noline/*.jpg -o data/indiana/datasets/abnormal_Z_320x320.h5 --resize 320,320
bin/preprocess.py hdf5_to_image -i data/indiana/datasets/abnormal_Z_320x320.h5 -d X --nrow 10 --ncol 4 -o tmp/preview/indiana_abnormal_Z_320x320.png
# merge multiple data sources
bin/preprocess.py merge_image_hdf5 -i data/anzhen/datasets/abnormal_Z_320x320.h5 \
    data/anzhen/datasets/normal_Z_320x320.h5 \
    data/indiana/datasets/normal_Z_320x320.h5 \
    data/indiana/datasets/abnormal_Z_320x320.h5 \
    -o data/merged/datasets/320x320.h5
```

## Download pretrained VGG-16 network:
Download from: [https://github.com/ShaoqingRen/faster_rcnn].
```bash
for model_name in 'ResNet50' 'VGG19' 'VGG16' 'InceptionV3';do
    bin/preprocess.py download_model --model-name $model_name --input-shape 320,320,3 -o models/imagenet/${model_name}_notop
done
```
```bash
bin/train.py classify_types -i data/anzhen/datasets/process_303x343.h5 \
    --indices-file output/classify_types/cv_indices \
    --indices-name 1/train -o output/classify_types/model
bin/test_model.py classify_types -i data/anzhen/datasets/process_303x343.h5 \
    --indices-file output/classify_types/cv_indices \
    --indices-name 1/test \
    --model-file output/classify_types/model \
    -o output/classify_types/predictions
```
## Detect edges using OpenCV
```bash
[ -d output/anzhen/detect_edges/process_606_686 ] || mkdir -p output/anzhen/detect_edges/process_606x686
{
for filename in $(ls data/anzhen/process_606_686/data | grep '.jpg$');do
    echo "data/anzhen/process_606_686/data/$filename output/anzhen/detect_edges/process_606x686/$filename"
done
} | xargs -L 1 -t -P 20 bin/detect_edges
# combine small images into a large one
montage $(find data/anzhen/process_606_686/data -type f | head -n 128) -tile 8x16 -geometry 128x128 tmp/original.jpg
montage $(find output/anzhen/detect_edges/process_606x686 -type f | head -n 128) -tile 8x16 -geometry 128x128 tmp/detect_edges.jpg
```
## Augment images
```bash
rm -r tmp/augmented_images
bin/preprocess.py augment_images -i data/anzhen/datasets/process_303x343.h5 -d X --n-images 25 --n-samples 10 \
    -o tmp/augmented_images
for sample in $(seq 0 9);do
    montage tmp/augmented_images/$sample/*.png -tile 5x5 -geometry 128x128 tmp/augmented_images/${sample}.png
done
```
## Convert traces to masks
```bash
# anzhen data (abnormal)
bin/preprocess.py trace_to_mask -i data/anzhen/abnormal_Z/heart_traces \
    --image-dir data/anzhen/process_512x624 \
    --resize 320,320 -o data/anzhen/abnormal_Z/heart_masks/320x320.h5
bin/preprocess.py hdf5_to_image -i data/anzhen/abnormal_Z/heart_masks/320x320.h5 \
    -d X --nrow 10 --ncol 4 -o tmp/preview/anzhen_heart_masks_320x320.png
# anzhen data (normal)
bin/preprocess.py trace_to_mask -i data/anzhen/normal_Z/heart_traces \
    --image-dir data/anzhen/normal_Z/noline \
    --resize 320,320 -o data/anzhen/normal_Z/heart_masks/320x320.h5
# merge multiple datasets
bin/preprocess.py merge_image_hdf5 -i data/anzhen/abnormal_Z/heart_masks/320x320.h5 \
    data/anzhen/normal_Z/heart_masks/320x320.h5 \
    -o data/merged/heart_masks/320x320.h5
```
