
# 🌡️ Weather Time Series Forecasting with Deep Learning (RNN, GRU, LSTM)

This project focuses on weather forecasting using a curated and optimized version of the Jena Climate dataset. The goal is to build and compare deep learning models (RNN, GRU, LSTM) to predict temperature (`T (degC)`) based on past weather observations. The models are trained on time series sequences and evaluated using multiple performance metrics.

---

## 📁 Dataset Description

The original dataset contains **15 atmospheric features** collected over time. For this project, I performed feature correlation analysis to select the most relevant features for temperature forecasting.

### ✅ Final Selected Features:
- `Date Time`
- `p (mbar)` — Air Pressure
- `rh (%)` — Relative Humidity
- `wv (m/s)` — Wind Velocity
- `max. wv (m/s)` — Max Wind Velocity
- `wd (deg)` — Wind Direction
- `T (degC)` — Temperature (Target Variable)

### 📊 Dataset Info:
- Shape: `(420,551 rows × 7 columns)`
- No missing values
- Sorted chronologically by `Date Time`
- Stored as a `.csv` file and shared publicly on Kaggle for use in time series and forecasting tasks

🔗 **[Kaggle Dataset Link]([https://your-kaggle-dataset-link](https://www.kaggle.com/datasets/samehraouf/jena-climate-time-series-2009-2017-dataset))**

---

## 🧪 Data Preparation & Processing

- Removed `Date Time` after sorting
- Used `StandardScaler` to normalize all features including the target
- Created sequences with a **lookback window of 72** (i.e. previous 12 hours of data)
- Split the data into:
  - **80% training set**
  - **20% testing set**

---

## 📊 Data Visualization

All visualizations were done using a **dark theme** with clear labels and contrast for presentation.

- ✅ 2D scatter plots using `matplotlib`
- ✅ 3D scatter plots using `matplotlib` (static) and `plotly` (dynamic)
- ✅ Seaborn pairplots for relationships among features
- ✅ Heatmap for feature correlation

---

## 🤖 Deep Learning Models

Three deep learning architectures were trained and compared. All models were trained for only **2 epochs** to simulate fast benchmarking.

### 🔹 RNN Model
```text
SimpleRNN → Dropout(0.3) → BatchNorm → SimpleRNN → Dense(16, relu) → Dense(1)
```

### 🔹 GRU Model
```text
Same structure as RNN, replacing SimpleRNN with GRU layers
```

### 🔹 LSTM Model
```text
Same structure as RNN, replacing SimpleRNN with LSTM layers
```

---

## 🧮 Evaluation Metrics

All models were evaluated using:
- R² Score
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)

### 📊 Results:

| Model | R² Score | MSE     | MAE     |
|-------|----------|---------|---------|
| RNN   | 0.9936   | 0.0059  | 0.0636  |
| GRU   | 0.9952   | 0.0044  | 0.0483  |
| LSTM  | 0.9928   | 0.0066  | 0.0627  |

📌 **GRU** showed the best performance, followed closely by RNN and LSTM.

---

## 📊 Results Visualization

- Bar chart to compare R², MSE, MAE across all models
- Subplots showing **actual vs predicted temperature** for each model
- Model architecture visualizations saved as PNGs using `plot_model()`

---

## 🧰 Tools & Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn, Plotly

---

## 📎 Project Links

- 📂 **[Kaggle Dataset](https://your-kaggle-dataset-link)**
- 📓 **[Notebook on Kaggle](http://your-kaggle-notebook-link)**

---

## 🚀 How to Run

```bash
1. Clone the repository
2. Install dependencies from requirements.txt
3. Run the notebook or script
```

---

## 🙌 Acknowledgements

- Jena Climate Dataset (original source)
- TensorFlow/Keras for deep learning framework
- Kaggle for dataset hosting

---

> Built and published with 💙 by [Your Name]
