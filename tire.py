import fastf1
import matplotlib.pyplot as plt
import os

# -------------------- SETUP CACHE --------------------
if not os.path.exists('cache'):
    os.makedirs('cache')

fastf1.Cache.enable_cache('cache')

# -------------------- LOAD SESSION --------------------
session = fastf1.get_session(2024, 'Monaco', 'R')
session.load()

# -------------------- SELECT DRIVER --------------------
driver = 'LEC'  # change to HAM, LEC, etc.
laps = session.laps.pick_driver(driver)

# -------------------- CLEAN DATA --------------------
laps = laps.pick_quicklaps().copy()
laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()

# -------------------- STINT DETECTION --------------------
laps['Stint'] = (laps['Compound'] != laps['Compound'].shift()).cumsum()

# -------------------- TIRE COLORS --------------------
compound_colors = {
    'SOFT': 'red',
    'MEDIUM': 'yellow',
    'HARD': 'white',
    'INTERMEDIATE': 'green',
    'WET': 'blue'
}

# -------------------- PLOT --------------------
plt.figure(figsize=(12, 6))

# F1-style dark blue background
ax = plt.gca()
fig = plt.gcf()

f1_bg = '#0B1D2A'   # deep navy blue
grid_color = '#1F3B4D'

fig.patch.set_facecolor(f1_bg)
ax.set_facecolor(f1_bg)

# Scatter plot per compound
for compound, group in laps.groupby('Compound'):
    plt.scatter(
        group['LapNumber'],
        group['LapTimeSeconds'],
        color=compound_colors.get(compound, 'gray'),
        label=compound,
        s=40
    )

# -------------------- TREND LINE --------------------
rolling = laps['LapTimeSeconds'].rolling(window=3).mean()
plt.plot(
    laps['LapNumber'],
    rolling,
    color='red',
    linewidth=2,
    label='Trend (3-lap avg)'
)

# -------------------- PIT STOP MARKERS --------------------
stint_changes = laps[laps['Stint'] != laps['Stint'].shift()]

for _, row in stint_changes.iterrows():
    plt.axvline(
        x=row['LapNumber'],
        color='white',
        linestyle='--',
        alpha=0.3
    )

# -------------------- LABELS --------------------
plt.xlabel('Lap Number', color='white')
plt.ylabel('Lap Time (seconds)', color='white')
plt.title(f'F1 Tire Degradation & Strategy ({driver})', color='white')

# Tick colors
ax.tick_params(colors='white')

# Grid
plt.grid(color=grid_color, alpha=0.3)

# Legend styling
legend = plt.legend()

# Set legend background color
legend.get_frame().set_facecolor('#0B1D2A')  # same dark blue
legend.get_frame().set_edgecolor('white')    # optional border

# Set text color to white
for text in legend.get_texts():
    text.set_color('white')

# -------------------- SAVE + SHOW --------------------
plt.savefig('f1_tire_analysis.png', dpi=300, facecolor=fig.get_facecolor())
plt.show()