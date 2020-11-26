# Mean Arterial Blood Pressure Morphology from Photoplethysmogram Raw Signal and Clinical Information Applying a Seq2seq Deep Learning Model with Attention

## 1) Pre-processing

```bash
1.1 Mount mimig.pg_dump
1.2 Run FILTER_sql_ABP_PPG.ipynb
1.3 Generate:
    * df_filtered_signals_icu_hadm_sex_age
    * parametros.json
    * parametros.mat
```    

## 2) Processing
```bash

2.1 Generate records:
  * 2.1.1: Unzip ABP_PPG.ziṕ
    or
  * 2.1.2: Run main.m
```

## 3) Deep_learning

```
This codes were runned into the Google Colaboratory enviroment.

Please pay attention to:

       "/content/gdrive/My\ Drive/ADJUST TO YOUR PATH"

and adjust it to your local path.
```
