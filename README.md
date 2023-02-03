# Two Stage Reweighting Survival Analysis
A python implementation of Two Stage Reweighting Survival Analysis (2SWSA) method in paper "Stabilising Job Survival Analysis for Disability Employment Services in Unseen Environments".

# Installation
For running 2SWSA:

* Python
* numpy
* pandas
* scipy
* sklearn
* survival
* torch

For running baseline methods:

* Cox proportional hazards, DeepSurv, Neural Multi-Task Logistic Regression, Random Survival Forest: pysurvival.
* Survival Support Vector Machine: sksurv.
* Accelerated Failure Time: lifelines.

# Infrastructure used to run experiments:
* OS: Red Hat Enterprise Linux, version 7.8.
* CPU: Intel(R) Xeon(R) Gold 6246 CPU @ 3.30GHz).
* RAM: 16 GB.

# Datasets

Telco Customer Churn (TEL). This dataset contains information about 7043 customers of a telecom company based in California with 19 features. The data is split into four sub datasets based on payment methods.

Mayo Clinic Primary Biliary Cirrhosis Data (PBC). This dataset contains information about 1945 patients with 16 attributes. The dataset is from the study of the progression of primary biliary cirrhosis. The data is split into four sub datasets based on histologic disease stage.

AIDS Clinical Trials Group Protocol 175 (ACG). This dataset consists of information from 2139 HIV infected patients. There are 25 features describing their characteristics, treatments they received, and outcomes. The data is split into five sub datasets based on patient age.

Kickstarter (KCS). This dataset contains information of 18093 crowdfunding  projects. The attributes in the kickstarter datasets include 56 features such as project goal amount, duration, textual content, etc. Each project in the kickstarter data is tracked over a period of time until either its goal date is reached or it obtains the goal amount. The data is split into four sub datasets based on extracting time.

Human resource (HMR). This dataset contains information of 15000 people with 10 features describing the characteristics of employees and their employment time. The data is split into nine sub datasets based on employee departments.

Framingham Heart study (FRH). This dataset contains information about 4434  participants  with 19 attributes in the Framingham heart study. It studies epidemiology for the hypertensive and arteriosclerotic cardiovascular disease. The data is split into four sub datasets based on education.

Study to Understand Prognoses Preferences Outcomes and Risks of Treatment (SPT). This is a public dataset introduced in a survival time study of seriously-ill hospitalised adults. After processing, the final dataset has 7856 samples with 43 attributes. The data is split into eight sub datasets based on disease type. 


# Usage

To run 2SWSA: execute command "python do_exp.py dataset 2swsa", where dataset is actg175, telco, pbc, hr, framingham or kickstarter. For example, to run experiments with the dataset of AIDS Clinical Trials Group Protocol 175, run "python do_exp.py actg175 2swsa".
