## Plot the model structures
```bash
for model_name in 'ResNet50' 'VGG19' 'VGG16' 'InceptionV3';do
    bin/analyze.py plot_model -m models/imagenet/${model_name}_notop -o output/analysis/plot_model/${model_name}_notop.png
done
```
## Visualize VGG16 networks
```bash
for layer_name in block1_conv1 block1_conv2 \
    block2_conv1 block2_conv2 \
    block3_conv1 block3_conv2 block3_conv3 \
    block4_conv1 block4_conv2 block4_conv3 \
    block5_conv1 block5_conv2 block5_conv3;do

    i_block=$(echo $layer_name | sed 's/block\([0-9]\)/\1/')
    if [ "i_block" = 1 ];then tile=4x16
    elif [ "i_block" = 2 ];then tile=8x16
    elif [ "i_block" = 3 ];then tile=32x16
    elif [ "i_block" = 4 ];then tile=32x16
    elif [ "i_block" = 5 ];then tile=32x16
    fi
    bin/visualize_networks.py conv_output --model-file models/imagenet/VGG16_notop \
        --layer-name $layer_name \
        -i data/anzhen/datasets/process_303x343.h5 \
        -n 10 -d X \
        -o tmp/visualize_networks/VGG16/$layer_name
    combined_dir=tmp/visualize_networks/VGG16/$layer_name/combined
    [ -d "$combined_dir" ] || mkdir -p "$combined_dir"
    for sample in $(seq 10);do
        montage tmp/visualize_networks/VGG16/$layer_name/$sample/*.png -tile $tile -geometry 100x100 $combined_dir/${sample}.png
    done
done
```

## Relevance analysis
```bash
classes=normal_anzhen,VSD
classes=normal_indiana,VSD
classes=normal_anzhen,VSD,ASD
classes=normal_indiana,VSD,ASD
method=deep_taylor
method=sensitivity
model=vgg16
mode=vgg16-mask
bin/analyze.py analyze_relevance --method $method \
    -m output/classify_diseases/$model/$classes/cv/0/model \
    -i data/merged/datasets/320x320.h5 --input-dataset X \
    -y output/classify_diseases/$model/$classes/targets --target-dataset y \
    --image-id-file output/classify_diseases/$model/$classes/cv_split \
    --image-id-dataset /0/test \
    -o output/classify_diseases/$model/$classes/cv/0/$method
bin/preprocess.py hdf5_to_image -i output/classify_diseases/$model/$classes/cv/0/$method \
    -d X --nrow 20 --ncol 10 --resize 128,128 \
    -o output/classify_diseases/$model/$classes/cv/0/preview/${method}.X.png
bin/preprocess.py hdf5_to_image -i output/classify_diseases/$model/$classes/cv/0/$method \
    -d relevance_map --nrow 20 --ncol 10 --resize 128,128 \
    -o output/classify_diseases/$model/$classes/cv/0/preview/${method}.relevance_map.png
```
