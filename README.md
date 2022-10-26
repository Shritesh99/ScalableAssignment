# ScalableAssignment

## Scalable Computing Project 2

### Train and test on local

Generating the training data

```
./generate.py --count 256 --output-dir train

```

Generating the val data

```
./generate.py --count 16 --output-dir val
```

Training the data

```
./train.py --batch-size 32 --epochs 100 --output-model model --train-dataset train --validate-dataset val
```

Uncompressing the files

```
rm -r output.csv
tar xf images.tar
```

Prediction on local

```
./classify.py --model-name model --captcha-dir images  --captcha-csv images.csv --output output.csv
```

Generating the lite model

```
./generate_lite_model.py --model-name model-predict.h5 --output model-lite
```

### Test on PI

Uncompressing the files

```
tar xf images.tar
```

Prediction on pi

```
./predict_lite.py --model model-lite.h5 --captcha-dir images --captcha-csv images.csv --output output_lite.csv
```
