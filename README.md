# Fetal Health Classification from Cardiotocogram Data

## Project Description

This project focuses on classifying fetal health using Cardiotocogram (CTG) data. The goal is to build a multiclass classification model that can predict fetal health states to help prevent maternal and neonatal mortality. By analyzing CTG features, the model aims to categorize fetal health into three distinct classes: Normal, Suspect, and Pathological.

## Dataset

The dataset consists of 2,126 records from CTG exams, with each record containing various features related to fetal heart rate and other vital signs. These features have been classified into three categories by expert obstetricians:

- **Normal** (Tagged as 1)
- **Suspect** (Tagged as 2)
- **Pathological** (Tagged as 3)

### Features

- **Baseline Value:** FHR baseline (beats per minute)
- **Accelerations:** Number of accelerations per second
- **Fetal Movement:** Number of fetal movements per second
- **Uterine Contractions:** Number of uterine contractions per second
- **Light Decelerations:** Number of light decelerations per second
- **Severe Decelerations:** Number of severe decelerations per second
- **Prolonged Decelerations:** Number of prolonged decelerations per second
- **Abnormal Short Term Variability:** Percentage of time with abnormal short-term variability
- **Mean Value of Short Term Variability:** Mean value of short-term variability
- **Percentage of Time with Abnormal Long Term Variability:** Percentage of time with abnormal long-term variability
- **Mean Value of Long Term Variability:** Mean value of long-term variability
- **Histogram Width:** Width of FHR histogram
- **Histogram Min:** Minimum (low frequency) of FHR histogram
- **Histogram Max:** Maximum (high frequency) of FHR histogram
- **Histogram Number of Peaks:** Number of histogram peaks
- **Histogram Number of Zeroes:** Number of histogram zeros
- **Histogram Mode:** Histogram mode
- **Histogram Mean:** Histogram mean
- **Histogram Median:** Histogram median
- **Histogram Variance:** Histogram variance
- **Histogram Tendency:** Histogram tendency

## Context

Reducing child and maternal mortality is a key global health priority and is a central focus of several United Nations Sustainable Development Goals. The UN aims to end preventable deaths of newborns and children under five by 2030 and to reduce under-five mortality rates to at least 25 per 1,000 live births. Maternal mortality remains a significant issue, particularly in low-resource settings.

Cardiotocograms (CTGs) are a cost-effective and accessible method for assessing fetal health, allowing healthcare professionals to intervene promptly and improve outcomes.

## Methodology

1. **Data Preprocessing:** Clean and prepare the CTG data for modeling.
2. **Feature Engineering:** Extract and engineer features relevant to fetal health classification.
3. **Model Building:** Develop and train a multiclass classification model to predict fetal health states.
4. **Evaluation:** Assess the model's performance and accuracy in predicting fetal health conditions.

## Acknowledgements

If you use this dataset in your research, please cite:

Ayres de Campos, D., et al. (2000). SisPorto 2.0: A Program for Automated Analysis of Cardiotocograms. *J Matern Fetal Med*, 5:311-318. [Link](http://example.com)

## License

The dataset license was not specified, but access is public with citation required.
