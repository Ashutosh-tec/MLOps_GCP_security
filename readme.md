RIS Classification Pipeline

This project builds and tests a Decision Tree classifier on the IRIS dataset, with MLOps features including:

- Data poisoning simulation
- GitHub Actions for CI/CD
- Unit testing with `unittest`
- CML reports on Pull Requests

---

## 📁 Project Structure

```
.
├── data/                     # Raw or poisoned IRIS data
├── models/                  # Saved model files
├── train.py                 # Model training script
├── test.py                  # Unit tests for the trained model
├── poison_data.py           # Function to simulate data poisoning
├── metrics.txt              # Output metrics used by CML
└── .github/
    └── workflows/
        └── sanity_check.yaml   # GitHub Actions CI workflow
```

---

## ⚙️ Setup

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Training the Model

```bash
python train.py
```

- Trains a Decision Tree on clean or poisoned IRIS data
- Saves model as `decision_tree_model.joblib`

---

## ✅ Run Unit Tests

```bash
python -m unittest test.py -v
```

- Tests model predictions and input schema

---

## 🧪 Data Poisoning

Poison IRIS data using:

```python
from poison_data import poison_data
data = poison_data(data, percent=0.1, noise_type='random')
```

- Simulates 5%, 10%, or 50% data poisoning attacks.

---

## 🔁 GitHub Actions CI

CI runs on:
- Push to `main` or `dev`
- Pull requests to `main`
- Manual dispatch

Workflow steps:
```yaml
- Checkout code
- Install dependencies
- Train model (train.py)
- Run tests (test.py)
- Create CML report (test_output.txt, metrics.txt)
```

Trigger manually if needed:
```bash
git commit --allow-empty -m "Trigger CI"
git push origin main
```

---

## 📄 License

MIT License
