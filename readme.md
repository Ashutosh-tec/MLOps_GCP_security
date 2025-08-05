# IRIS Classification Pipeline with DVC

This project demonstrates how to build a simple machine learning pipeline for classifying iris species using a Decision Tree classifier. It integrates **DVC (Data Version Control)** to manage data and model versions efficiently.

---

## 🔍 Problem Statement

Incorporate **DVC** to version control the local data and model artifacts within an existing ML pipeline for IRIS dataset classification.

---


---


## Initialize DVC (if not already initialized)  
```
dvc init
```

## Restore data from DVC
```
dvc checkout iris.csv.dvc
```


---

## 📌 Key Learnings

1. **Data Versioning**  
Used `dvc add` to track the `iris.csv` dataset and manage different versions across Git commits and tags.

2. **Model Reproducibility**  
Trained and saved the `model.joblib` file, tracked with DVC to ensure model version consistency.

3. **Tag-based Rollback**  
Used Git tags (`v0.1`, `v0.2`) along with DVC to switch between original and tweaked versions of the dataset for easy experimentation.

---

## 🏷️ Version History

- `v0.1`: Initial version with original dataset (151 rows).
- `v0.2`: Updated dataset with tweaks (177 rows)

---

## ✅ Result

The pipeline with DVC integration helps efficiently manage data/model files and supports reproducibility and collaboration in ML workflows.

---

