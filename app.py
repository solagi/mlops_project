import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


# helper function to read prediction output file (containing Adurino CLI output) into a list
def get_prediction_values(filepath):
    values = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip() == '#Regression results:':
            # Look for the next line containing 'value:'
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith('value:'):
                    try:
                        val = float(next_line.split(':')[1].strip())
                        values.append(val)
                    except Exception:
                        pass
    return values

daily_light = get_prediction_values(filepath="log.txt")

# Title
st.title("💡 Light Intensity Visualizer")


# Display current value
avg_intensity = np.mean(daily_light) if daily_light else 0
st.metric("Current AVG Light Level", f"{avg_intensity:.2f} lux")

# Generate sample daily data
st.subheader("Sample Daily Pattern")
# daily_light = get_prediction_values(filepath="log.txt")
hours = np.arange(0,len(daily_light))#np.arange(0, 24)

# Line chart
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(hours, daily_light, color='orange', linewidth=2)
ax2.fill_between(hours, daily_light, alpha=0.3, color='yellow')
ax2.set_xlabel('Time datapoint (seconds)')
ax2.set_ylabel('Light Intensity (lux)')
ax2.set_xlim(0, 24)
ax2.set_ylim(30, 60)  # Adjusted y-axis range
ax2.grid(True, alpha=0.3)
st.pyplot(fig2)


