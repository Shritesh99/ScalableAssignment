# ScalableAssignment

## Scalable Computing Project 2

### Train and test on local

```
./generate.py --count 256 --output-dir train

```

```
./generate.py --count 16 --output-dir val
```

```
./train.py --batch-size 32 --epochs 100 --output-model model --train-dataset train --validate-dataset val
```

```
rm -r output.csv
tar xf images.tar
```

```
./classify.py --model-name model --captcha-dir images  --captcha-csv images.csv --output output.csv
```

```
./generate_lite_model.py --model-name model --output model.lite
```

### Test on PI

```
./predict_lite.py --model-name ./model-lite.h5 --captcha-dir images --captcha-csv images.csv --output output_lite.csv
```
