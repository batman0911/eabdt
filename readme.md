# Sentiment Analysis

## Data

- Dataset: https://1drv.ms/f/s!AhvVyQ1gW2O9gbVOpe6CyYYp0CYvag?e=IcxK4M

## Batch processing

- Batch processing `python/vectorize.ipynb`
- Validation (splitted file and embedded maxtrices) `python/validation.ipynb`

## Embedding

```sh
python python/vectorize.py {dataset} {subset} {numbatch}
```

where 
- `dataset` : mix, All_Beauty, Movies_and_TV
- `subset` : training_set, testing_set
- `numbatch` : number of batchs to process (start from batch 0)

## Classification

- `xgboost`: `python/classification.py`
- `SGD`, `Navie-Bayes`, `Logistic-Regression`: `python/classifier.ipynb`
- `Deep-learning`: `python/simple_softmax.ipynb`, `python/softmax.ipynb`

## Model loader

- Load model and print metrics: `python/model_loader.ipynb`

## Distributed Training

- project `spark_classification` (with `scala`)
