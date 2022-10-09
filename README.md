# ScalableAssignment

Scalable Computing Toy Project

```
./generate.py --width 128 --height 64 --length 5 --symbols symbols.txt --count 128000 --output-dir test

```
```
./generate.py --width 128 --height 64 --length 5 --symbols symbols.txt --count 12800 --output-dir val
```

```
./train.py --width 128 --height 64 --length 5 --symbols symbols.txt --batch-size 32 --epochs 100 --output-model test.h5 --train-dataset test --validate-dataset val
```

```
./classify.py --model-name test.h5 --captcha-dir jamulkas-project1 --output result.csv --symbols symbols.txt
```
