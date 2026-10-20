import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

TICKER = "ORK.OL"
START = "2020-01-01"
SLUTT = "2026-01-01"

# Hent ekstra historikk slik at indikatoren er klar når perioden starter.
data = yf.download(
    TICKER,
    start="2019-07-01",
    end=SLUTT,
    auto_adjust=True,
    multi_level_index=False,
    progress=False
)

if data.empty:
    raise ValueError("Fant ingen kursdata.")

data = data.dropna(subset=["Close", "Volume"]).copy()

# On-Balance Volume (OBV):
# Dagens volum legges til når kursen stiger, trekkes fra når den faller,
# og OBV er uendret når sluttkursen er lik gårsdagens.
retning = np.sign(data["Close"].diff()).fillna(0)
data["OBV"] = (retning * data["Volume"]).cumsum()

# Signallinje: 20-dagers glidende snitt av OBV.
data["OBV-signal"] = data["OBV"].rolling(20, min_periods=20).mean()

# Behold samme sammenligningsperiode som i MACD-programmet.
data = data.loc[START:].copy()

if data.empty:
    raise ValueError("Fant ingen kursdata i perioden som skal vises.")

data["Daglig avkastning"] = data["Close"].pct_change().fillna(0.0)

# 1 = eier aksjen, 0 = står i kontanter.
# Eier aksjen når OBV ligger over signallinjen (kjøpepresset tiltar).
# Dagens signal gjelder først neste handelsdag.
data["OBV-posisjon"] = (
    (data["OBV"] > data["OBV-signal"])
    .shift(1, fill_value=False)
    .astype(int)
)

data["OBV daglig"] = (
    data["OBV-posisjon"] * data["Daglig avkastning"]
)
data["Kjøp og hold daglig"] = data["Daglig avkastning"]

data["OBV samlet"] = (
    (1 + data["OBV daglig"]).cumprod() - 1
) * 100
data["Kjøp og hold samlet"] = (
    (1 + data["Kjøp og hold daglig"]).cumprod() - 1
) * 100

print(f"OBV-strategi: {data['OBV samlet'].iloc[-1]:.1f} %")
print(f"Kjøp og hold: {data['Kjøp og hold samlet'].iloc[-1]:.1f} %")

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    data.index,
    data["OBV samlet"],
    color="#2a78d6",
    label="OBV-strategi"
)
ax.plot(
    data.index,
    data["Kjøp og hold samlet"],
    color="#eb6834",
    label="Kjøp og hold"
)

ax.axhline(0, color="gray", linewidth=1)
ax.set_title("Orkla-kurs: sammenligning fra 2020 til 2025")
ax.set_xlabel("Dato")
ax.set_ylabel("Samlet avkastning (%)")
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()