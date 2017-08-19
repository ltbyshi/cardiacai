## Summarize cross-validation results
```bash
model=vgg16
model=vgg16_mask_unet_vgg16
for target in normal_anzhen,VSD,ASD normal_anzhen,VSD normal_indiana,VSD normal_indiana,VSD,ASD;do
    bin/report.py summarize_cv -i output/classify_diseases/$model/$target \
        -a "method=$model" -a "target=$target" -o report/classify_diseases/${model}_${target}.txt
done
```