# Laptop-Price-Prediction-Model
A machine learning model to predict laptop prices based on specifications like brand, CPU, RAM, storage, GPU, screen features, and OS. Built using Python, Pandas, scikit-learn, and Matplotlib.

---

## 🚀 Features

- Preprocessing & feature extraction (e.g. screen resolution, CPU type)
- EDA using Seaborn & Matplotlib
- Trained regression model (e.g. Random Forest)
- Exported pipeline (`pipe.pkl`) for predictions

---

## 🗂️ Files in This Repo

- `laptop_data.csv` – Raw dataset  
- `Laptop_Price_Prediction.ipynb` – Full notebook (EDA to model)  
- `df.pkl` – Cleaned dataset  
- `pipe.pkl` – Trained model pipeline  
- `requirements.txt` – Dependencies list

---

## 📦 Setup Instructions

```bash
git clone https://github.com/VedAditya-10/Laptop-Price-Prediction-Model.git
cd Laptop-Price-Prediction-Model
pip install -r requirements.txt
```

---

## 🧪 Predicting Price

```python
import pickle
import pandas as pd

pipe = pickle.load(open('pipe.pkl', 'rb'))

sample = {
    'Company': 'Dell',
    'TypeName': 'Ultrabook',
    'Ram': 8,
    'Weight': 1.2,
    'Touchscreen': 0,
    'Ips': 1,
    'ScreenSize': 13.3,
    'Resolution': '1920x1080',
    'Cpu': 'Intel Core i5',
    'HDD': 0,
    'SSD': 256,
    'Gpu': 'Intel HD Graphics 620',
    'OpSys': 'Windows 10'
}

df = pd.DataFrame([sample])
price = pipe.predict(df)[0]
print(f"Predicted Price: ₹{round(price)}")
```

---

## 🛠️ Tech Stack

- Python, Pandas, NumPy
- scikit-learn
- Matplotlib, Seaborn
- Pickle

---

## 📬 Contact

For queries or contributions, connect via GitHub: [VedAditya-10](https://github.com/VedAditya-10)
