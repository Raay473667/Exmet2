import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit
#from scipy.integrate import trapz

file_path = 'data/Zinek.csv'

data = pd.read_csv(file_path, sep=';')

fig, ax = plt.subplots()
# Extract column headers
F_head = data.columns[0]
h_head = data.columns[1]
t_head = data.columns[2]
HMu_head = data.columns[3]
hcorr_head = data.columns[4]
HM_head = data.columns[5]

# Extract data from the first and second columns
F = data[F_head]
h = data[h_head]
t = data[t_head]
HMu = data[HMu_head]
hcorr = data[hcorr_head]
HM = data[HM_head]

# Create a plot
plt.plot(h, F)

# Label the axes
plt.xlabel(h_head)
plt.ylabel(F_head)

# Display the plot
#plt.show()
with PdfPages('plots/F_h.pdf') as pdf:
        pdf.savefig(fig, bbox_inches='tight')
plt.close()

plt.plot(hcorr, F)
plt.xlabel(hcorr_head)
plt.ylabel(F_head)
#plt.show()
with PdfPages('plots/F_hcorr.pdf') as pdf:
        pdf.savefig(fig, bbox_inches='tight')
plt.close()

# Define the index at which you want to split the data
split_index = 199  # Replace 50 with the index you want to use for splitting

# Split the DataFrame into two new DataFrames
data1 = data.iloc[:split_index]
split_index = split_index + 11
data2 = data.iloc[split_index:]

# Split the x_data column into two new columns
F1 = data1[data1.columns[0]]
F2 = data2[data2.columns[0]]
h1 = data1[data1.columns[1]]
h2 = data2[data2.columns[1]]
t1 = data1[data1.columns[2]]
t2 = data2[data2.columns[2]]
HMu1 = data1[data1.columns[3]]
HMu2 = data2[data2.columns[3]]
hcorr1 = data1[data1.columns[4]]
hcorr2 = data2[data2.columns[4]]
HM1 = data1[data1.columns[5]]
HM2 = data2[data2.columns[5]]

h_min = 7
h_max = 60
hcorr2_filtered = hcorr2.iloc[h_min:h_max + 1]
F2_filtered = F2.iloc[h_min:h_max + 1]

# Extract the filtered x and y values
#filtered_x_data = filtered_data[filtered_data.columns[0]]
#filtered_y_data = filtered_data[filtered_data.columns[1]]
a, b = np.polyfit(hcorr2_filtered, F2_filtered, 1)
hr = -b/a

fig, ax = plt.subplots()
ax.plot(hcorr1, F1)
ax.plot(hcorr2, F2)
ax.plot(hcorr2_filtered, a*hcorr2_filtered+b, '--')
plt.xlabel(hcorr_head)
plt.ylabel(F_head)





print(f'hr = {hr:.2f}')
hcorr2_hr = hcorr2[(hcorr2 >= hr)]
hcorr2_hr = hcorr2_hr.iloc[::-1]
F2_hr = F2[(hcorr2 >= hr)]
F2_hr = F2_hr.iloc[::-1]
#print(hcorr1)
#print(F1)
#print(hcorr2_hr)
#print(F2_hr)
deformationWork_tot = np.trapz(F1, hcorr1)
deformationWork_elast = np.trapz(F2_hr, hcorr2_hr)
deformationWork_irr = deformationWork_tot - deformationWork_elast
print(f'deformationWork_tot = {deformationWork_tot:.1f}')
print(f'deformationWork_elast = {deformationWork_elast:.1f}')
print(f'deformationWork_irr = {deformationWork_irr:.1f}')

Fmax = F.max() / 1000  # [N]
hmax = h.max() / 1e6   # [m]
K = 26.43
Martens = Fmax / hmax**2 / K
#print(Fmax, hmax, Martens)
print(f'Martens = {Martens:.2e}')

def OP(h, B, m):
    hp = h.iloc[-1]
    return B * (h - hp)**m

# Perform the curve fitting
initial_guess = [1, 2]  # Provide an initial guess for the parameters B and m
optimal_params, _ = curve_fit(OP, hcorr2, F2, p0=initial_guess)

# Print the optimal parameters
B_opt, m_opt = optimal_params

print("Optimal parameters:")
print("B =", B_opt)
print("m =", m_opt)
ax.plot(hcorr2, OP(hcorr2, B_opt, m_opt))


with PdfPages('plots/F_hcorr_split.pdf') as pdf:
        pdf.savefig(fig, bbox_inches='tight')
#plt.show()
plt.close()