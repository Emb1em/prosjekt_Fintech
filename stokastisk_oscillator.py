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
 
data = data.dropna(subset=["High", "Low", "Close"]).copy()
 
# Stokastisk oscillator (14, 3, 3):
# Rå %K viser hvor sluttkursen ligger i de siste 14 dagenes kursintervall.
laveste_14 = data["Low"].rolling(14, min_periods=14).min()
hoyeste_14 = data["High"].rolling(14, min_periods=14).max()
spenn = hoyeste_14 - laveste_14
 
ra_k = 100 * (data["Close"] - laveste_14) / spenn.where(spenn.ne(0))
data["%K"] = ra_k.rolling(3, min_periods=3).mean()
data["%D"] = data["%K"].rolling(3, min_periods=3).mean()
 
# Behold samme sammenligningsperiode som i MACD-programmet.
data = data.loc[START:].copy()
 
if data.empty:
    raise ValueError("Fant ingen kursdata i perioden som skal vises.")
 
data["Daglig avkastning"] = data["Close"].pct_change().fillna(0.0)
 
# 1 = eier aksjen, 0 = står i kontanter.
# Dagens signal gjelder først neste handelsdag.
data["Stokastisk-posisjon"] = (
    (data["%K"] > data["%D"])
    .shift(1, fill_value=False)
    .astype(int)
)
 
data["Stokastisk daglig"] = (
    data["Stokastisk-posisjon"] * data["Daglig avkastning"]
)
data["Kjøp og hold daglig"] = data["Daglig avkastning"]
 
data["Stokastisk samlet"] = (
    (1 + data["Stokastisk daglig"]).cumprod() - 1
) * 100
data["Kjøp og hold samlet"] = (
    (1 + data["Kjøp og hold daglig"]).cumprod() - 1
) * 100
 
print(f"Stokastisk strategi: {data['Stokastisk samlet'].iloc[-1]:.1f} %")
print(f"Kjøp og hold:         {data['Kjøp og hold samlet'].iloc[-1]:.1f} %")
 
fig, ax = plt.subplots(figsize=(12, 6))
 
ax.plot(
    data.index,
    data["Stokastisk samlet"],
    color="#2a78d6",
    label="Stokastisk strategi"
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