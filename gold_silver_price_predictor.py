import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tkinter as tk

# Download 5 Years Data
gold = yf.download("GC=F", period="5y")
silver = yf.download("SI=F", period="5y")

def prepare_model(data):
    df = pd.DataFrame()
    df["Close"] = data["Close"]
    df["Return"] = df["Close"].pct_change()
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
    df.dropna(inplace=True)

    X = df[["Return", "MA5", "MA10"]]
    y = df["Target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    latest = X_scaled[-1].reshape(1, -1)
    prediction = model.predict(latest)[0]
    probability = model.predict_proba(latest)[0][prediction]

    return df, prediction, probability

gold_df, gold_pred, gold_prob = prepare_model(gold)
silver_df, silver_pred, silver_prob = prepare_model(silver)

gold_last = gold_df.tail(30)
silver_last = silver_df.tail(30)

plt.style.use("dark_background")
plt.figure(figsize=(14,7))

plt.plot(gold_last["Close"], linewidth=3)
plt.plot(silver_last["Close"], linewidth=3)

plt.title("✨ Gold & Silver Analysis (Last 1 Month) ✨", fontsize=18)
plt.legend(["Gold", "Silver"])
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Fun Popup
root = tk.Tk()
root.title("🚀 AI Metal Trading Signals")
root.geometry("550x350")
root.resizable(False, False)

gold_signal = "BUY 📈" if gold_pred == 1 else "DO NOT BUY 📉"
silver_signal = "BUY 📈" if silver_pred == 1 else "DO NOT BUY 📉"

message = f"""
💎 GOLD SIGNAL:
{gold_signal}
Confidence: {round(gold_prob*100,2)}%

🥈 SILVER SIGNAL:
{silver_signal}
Confidence: {round(silver_prob*100,2)}%
"""

bg_color = "#14532d" if gold_pred == 1 and silver_pred == 1 else "#7f1d1d"

root.configure(bg=bg_color)

label = tk.Label(
    root,
    text=message,
    font=("Helvetica", 16, "bold"),
    fg="white",
    bg=bg_color,
    justify="center"
)
label.pack(expand=True)

button = tk.Button(
    root,
    text="Close",
    font=("Helvetica", 14, "bold"),
    bg="black",
    fg="white",
    padx=20,
    pady=5,
    command=root.destroy
)
button.pack(pady=15)

root.mainloop()