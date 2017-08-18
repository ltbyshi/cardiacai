## Classify diseases
Set environment variables:
```bash
classes=normal_anzhen,VSD
classes=normal_indiana,VSD
classes=normal_indiana,VSD,ASD
classes=normal_anzhen,VSD,ASD
model=vgg16
```
Run the training pipeline:
```bash
bin/preprocess.py create_dataset -i data/sample_info.xlsx \
    --classes ${classes} \
    --task classify_diseases \
    -o output/classify_diseases/${model}/${classes}/targets
bin/preprocess.py cv_split -i output/classify_diseases/${model}/${classes}/targets \
    -k 10 --valid-size 0.1 \
    -o output/classify_diseases/${model}/${classes}/cv_split
for cv_fold in $(seq 2 10);do
bin/train.py classify_diseases -i data/merged/datasets/320x320.h5 \
    -y output/classify_diseases/${model}/${classes}/targets \
    -m models/imagenet/VGG16_notop \
    --cv-split-file output/classify_diseases/${model}/${classes}/cv_split --cv-fold $cv_fold \
    --batch-size 32 --fine-tune \
    -o output/classify_diseases/${model}/${classes}/cv/$cv_fold \
    --epochs 10
done
```
## Segment images
```bash
bin/preprocess.py create_dataset -i data/sample_info.xlsx \
    --task segment --data-source anzhen \
    -o output/segment/anzhen/image_ids
bin/preprocess.py cv_split -i output/segment/anzhen/image_ids \
    -k 10 --valid-size 0.1 \
    -o output/segment/anzhen/cv_split
bin/train.py segment -i data/merged/datasets/320x320.h5 \
    -y data/merged/heart_masks/320x320.h5 \
    --cv-split-file output/segment/anzhen/cv_split --cv-fold 0 \
    --batch-size 32 --epochs 20 \
    -o output/segment/anzhen/cv/0
bin/predict.py segment -i data/merged/datasets/320x320.h5 \
    --model-file output/segment/anzhen/model \
    --image-id-file output/segment/anzhen/cv_split \
    --image-id-dataset /0/test \
    --mix \
    -o output/segment/anzhen/cv/0/predictions
bin/preprocess.py hdf5_to_image -i output/segment/anzhen/predictions \
    -d X --nrow 50 --ncol 5 \
    -o output/segment/anzhen/preview/predictions.png
```

### apply anzhen model to indiana data
```bash
bin/preprocess.py create_dataset -i data/sample_info.xlsx \
    --task segment --data-source indiana \
    -o output/segment/indiana/image_ids
bin/predict.py segment -i data/merged/datasets/320x320.h5 \
    --model-file output/segment/anzhen/cv/0/model \
    --image-id-file data/segment/indiana/image_ids \
    --image-id-dataset image_id \
    --mix \
    -o output/segment/anzhen/cv/0/predictions.indiana
bin/preprocess.py hdf5_to_image -i output/segment/anzhen/cv/0/predictions.indiana \
    -d X --nrow 50 --ncol 5 \
    -o output/segment/anzhen/cv/0/preview/predictions.indiana.png
```
### Apply anzhen model to all data
```bash
model_dir=output/segment/anzhen/cv/0
# mix mask with original images
bin/predict.py segment -i data/merged/datasets/320x320.h5 \
    --model-file $model_dir/model \
    --image-id-file data/merged/datasets/320x320.h5 \
    --image-id-dataset image_id \
    --mix \
    -o $model_dir/predictions.all.mix
bin/preprocess.py hdf5_to_image -i $model_dir/predictions.all.mix \
    -d X --nrow 20 --ncol 10 --resize 128,128 \
    -o $model_dir/preview/predictions.all.mix.png
# only mask
bin/predict.py segment -i data/merged/datasets/320x320.h5 \
    --model-file $model_dir/model \
    --image-id-file data/merged/datasets/320x320.h5 \
    --image-id-dataset image_id \
    -o $model_dir/predictions.all
bin/preprocess.py hdf5_to_image -i $model_dir/predictions.all \
    -d X --nrow 20 --ncol 10 --resize 128,128 \
    -o $model_dir/preview/predictions.all.png
```
### Classify diseases using segmentation result
```bash
model=vgg16-mask
output_dir=output/classify_diseases/vgg16-mask/${classes}
[ -d "$output_dir" ] || mkdir -p "$output_dir"
cp output/classify_diseases/vgg16/${classes}/cv_split $output_dir/
cp output/classify_diseases/vgg16/${classes}/targets $output_dir/
bin/train.py classify_diseases -i data/merged/datasets/320x320.h5 \
    -y output/classify_diseases/${model}/${classes}/targets \
    --mask-file output/segment/anzhen/cv/0/predictions.all \
    -m models/imagenet/VGG16_notop \
    --cv-split-file output/classify_diseases/${model}/${classes}/cv_split --cv-fold 0 \
    --batch-size 32 --fine-tune \
    -o output/classify_diseases/${model}/${classes}/cv/0 \
    --epochs 10
```
