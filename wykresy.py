
#WYKRES 1



import matplotlib.pyplot as plt

# Dane do wykresu
scenarios = ["Konserwatywny\n25 000 paczek",
             "Typowy\n30 000 paczek",
             "Optymistyczny\n40 000 paczek"]
profits = [30000, 36000, 48000]  # zysk netto w PLN
profits_thousands = [p / 1000 for p in profits]  # w tys. zł

# Definiujemy kolory – środkowy słupek pomarańczowy, pozostałe szare
colors = ["gray", "orange", "gray"]

# Tworzymy wykres
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(scenarios, profits_thousands, color=colors)

# Etykiety osi i tytuł
ax.set_ylabel("Zysk netto (tys. zł rocznie)")
ax.set_title("Estymowany roczny zysk netto z pojedynczego Paczkomatu")

# Dodajemy wartości nad słupkami
for i, (bar, value) in enumerate(zip(bars, profits_thousands)):
    if i == 1:  # środkowy słupek
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.5,
                f"{value:.0f}", ha="center", va="bottom", fontweight="bold")
    else:
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.5,
                f"{value:.0f}", ha="center", va="bottom")

plt.tight_layout()
plt.show()





#WYKRES 2

import matplotlib.pyplot as plt
import numpy as np

# Dane liczby Paczkomatów (InPost, Polska) - wartości na koniec roku
years = np.array([2020, 2021, 2022, 2023, 2024, 2025])
lockers = np.array([12254, 16000, 19306, 21969, 25269, 28269])  # 2025 = plan

fig, ax = plt.subplots(figsize=(10, 6))

# Szare słupki, ostatni pomarańczowy
colors = ["gray"] * (len(years) - 1) + ["orange"]
bars = ax.bar(years, lockers, color=colors)

# Etykiety nad słupkami
for i, bar in enumerate(bars):
    height = bar.get_height()
    if i == len(bars) - 1:  # ostatni słupek
        ax.text(bar.get_x() + bar.get_width() / 2, height + 400, f"{height:,}", ha="center", va="bottom", fontweight="bold")
    else:
        ax.text(bar.get_x() + bar.get_width() / 2, height + 400, f"{height:,}", ha="center", va="bottom")

# Opisy osi i tytuł
ax.set_ylabel("Liczba Paczkomatów")
ax.set_title("Wzrost liczby Paczkomatów InPost w Polsce (2020–2025)")

# Usuwamy prawą oś (nie było twin axis)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)

plt.tight_layout()
plt.show()




#WYKRES 3


import matplotlib.pyplot as plt
import numpy as np

# Założenia bazowe
base_year = 2024
base_parcels = 709_000_000  # przesyłki InPost 2024 (w szt.)
profit_per_parcel = 1.2  # zysk netto w PLN na paczkę
years = np.arange(base_year, 2029)  # 2024–2028 włącznie

# Założenia wzrostu popytu (scenariusze)
growth_low = 0.08   # 8% r/r – dolny przedział
growth_mid = 0.12   # 12% r/r – scenariusz bazowy
growth_high = 0.16  # 16% r/r – górny przedział

# Funkcja do wyliczenia wolumenu i zysku
def forecast(parcels, growth):
    v = [parcels]
    for _ in range(1, len(years)):
        v.append(v[-1] * (1 + growth))
    return np.array(v)

parcels_low = forecast(base_parcels, growth_low)
parcels_mid = forecast(base_parcels, growth_mid)
parcels_high = forecast(base_parcels, growth_high)

# Zyski (w mld zł)
profits_low = parcels_low * profit_per_parcel / 1e9
profits_mid = parcels_mid * profit_per_parcel / 1e9
profits_high = parcels_high * profit_per_parcel / 1e9

# Rysujemy wykres z przedziałami ufności
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(years, profits_mid, label="Scenariusz bazowy")
ax.fill_between(years, profits_low, profits_high, alpha=0.3, label="Przedział ufności (8–16% wzrostu)")

ax.set_ylabel("Zysk netto (mld zł)")
ax.set_title("Prognoza rocznego zysku netto InPost\n(przy spełnieniu popytu na wysyłkę paczek)")
ax.set_xticks(years)
ax.grid(True, linestyle="--", linewidth=0.5)
ax.legend()

# Dodajemy etykiety punktowe
for x, y in zip(years, profits_mid):
    ax.text(x, y + 0.05, f"{y:.1f}", ha="center", va="bottom")

plt.tight_layout()
plt.show()
