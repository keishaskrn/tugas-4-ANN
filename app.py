from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("model_ann.h5", compile=False)

# Load dataset untuk scaler
df = pd.read_csv("golongan_darah.csv")
df.columns = df.columns.str.strip().str.lower()

# ambil data numerik
df = df.select_dtypes(include=[np.number])

X = df.iloc[:, 0].values.reshape(-1,1)
Y = df.iloc[:, -1].values.reshape(-1,1)

# scaler
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()

scaler_X.fit(X)
scaler_Y.fit(Y)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    tahun = float(request.form["tahun"])

    data = np.array([[tahun]])
    data_scaled = scaler_X.transform(data)

    pred_scaled = model.predict(data_scaled)
    hasil = scaler_Y.inverse_transform(pred_scaled)

    return render_template("hasil.html", hasil=int(hasil[0][0]))

if __name__ == "__main__":
    app.run(debug=True)