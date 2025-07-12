```
python main.py --name {name} --config_file {config.yaml} --gpu 0 --train
```


# Milano Datasets

## SIS-IN


### Forcasting

- Training
```
python main.py --name SMS-IN --config_file configs/sms_in.yaml --gpu 0 --train --mode predict
```

- Testing
```
python main.py --name SMS-IN --config_file configs/sms_in.yaml --gpu 0 --sample 1 --mode predict
```

- Testing with --missing_ratio
```
python main.py --name SMS-IN --config_file configs/sms_in.yaml --gpu 0 --sample 1 --mode predict --missing_ratio 0.8
```

### Imputation

- Training
```
python main.py --name SMS-IN --config_file configs/sms_in.yaml --gpu 0 --train --mode infill
```

- Testing
```
python main.py --name SMS-IN --config_file configs/sms_in.yaml --gpu 0 --sample 1 --mode infill --missing_ratio 0.2
```

### Generation

```
python main.py --name SMS-IN --config_file configs/sms_in.yaml --gpu 0
```

## SIS-OUT

### Forcasting

- Training
```
python main.py --name SMS-OUT --config_file configs/sms_out.yaml --gpu 1 --train --mode predict
```

test
```
python main.py --name SMS-OUT --config_file configs/sms_out.yaml --gpu 1 --sample 1 --mode predict
```

### Imputation

- Training
```
python main.py --name SMS-OUT --config_file configs/sms_out.yaml --gpu 1 --train --mode infill
```
- Testing
```
python main.py --name SMS-OUT --config_file configs/sms_out.yaml --gpu 1 --sample 1 --mode infill --missing_ratio 0.2
```


## Call-IN

### Forcasting

- Training
```
python main.py --name Call-IN --config_file configs/call_in.yaml --gpu 2 --train --mode predict
```

- Testing
test
```
python main.py --name Call-IN --config_file configs/call_in.yaml --gpu 2 --sample 1 --mode predict
```

### Imputation

- Training
```
python main.py --name Call-IN --config_file configs/call_in.yaml --gpu 2 --train --mode infill
```
- Testing
```
python main.py --name Call-IN --config_file configs/call_in.yaml --gpu 2 --sample 1 --mode infill --missing_ratio 0.2
```


## Call-OUT

### Forcasting

- Training
```
python main.py --name Call-OUT --config_file configs/call_out.yaml --gpu 3 --train --mode predict
```

- Testing
```
python main.py --name Call-OUT --config_file configs/call_out.yaml --gpu 3 --sample 1 --mode predict
```

### Imputation

- Training
```
python main.py --name Call-OUT --config_file configs/call_out.yaml --gpu 3 --train --mode infill
```
- Testing
```
python main.py --name Call-OUT --config_file configs/call_out.yaml --gpu 3 --sample 1 --mode infill --missing_ratio 0.2
```

## Internet

### Forcasting

- Training
```
python main.py --name Internet --config_file configs/internet.yaml --gpu 4 --train --mode predict
```

- Testing
```
python main.py --name Internet --config_file configs/internet.yaml --gpu 4 --sample 1 --mode predict
```

### Imputation

- Training
```
python main.py --name Internet  --config_file configs/internet.yaml --gpu 4 --train --mode infill
```
- Testing
```
python main.py --name Internet --config_file configs/internet.yaml --gpu 4 --sample 1 --mode infill --missing_ratio 0.2
```


## ETTh Datasets

test
```
python main.py --name ETTh --config_file Config/etth.yaml --gpu 0 --sample 1  --mode predict --pred_len 96
```