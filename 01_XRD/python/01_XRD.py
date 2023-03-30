import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
from functools import partial
from matplotlib.ticker import AutoMinorLocator 
import math
from PyPDF2 import PdfMerger
import scipy.signal as signal

def round_down_even(x):
    """
    Round x down so that the first decimal point is even.
    """
    y = math.floor(x * 10) / 10
    if int(y * 10) % 2 == 0:
        y += 0.1
    return y

def two_gaussians(x, a1, sigma1, a2, mu1, mu2):
    fwhm = 2*np.sqrt(2*np.log(2))*sigma1
    gauss1 = a1 * np.exp(-(x - mu1)**2 / (2 * fwhm**2))
    gauss2 = a2 * np.exp(-(x - mu2)**2 / (2 * fwhm**2))
    lorentz1 =  a1 * (sigma1 / 2)**2 / ((x - mu1)**2 + (sigma1 / 2)**2)
    lorentz2 =  a2 * (sigma1 / 2)**2 / ((x - mu2)**2 + (sigma1 / 2)**2)
    return lorentz1 + lorentz2
    return gauss1 + gauss2

def fit_gauss_two(x, y, bounds, p0):
    #fixed_gaussians = partial(two_gaussians, mu1=mu1, mu2=mu2)
    popt, pcov = curve_fit(two_gaussians, x, y, p0=p0, bounds=bounds, ftol=1e-10)
    return popt

def first_gaussian(x, a1, sigma1, mu1):
    fwhm = 2*np.sqrt(2*np.log(2))*sigma1
    gauss1 = a1 * np.exp(-(x - mu1)**2 / (2 * fwhm**2))
    lorentz1 =  a1 * (sigma1 / 2)**2 / ((x - mu1)**2 + (sigma1 / 2)**2)
    return lorentz1
    return gauss1

def second_gaussian(x, a2, sigma2, mu2):
    fwhm = 2*np.sqrt(2*np.log(2))*sigma2
    gauss2 = a2 * np.exp(-(x - mu2)**2 / (2 * fwhm**2))
    lorentz2 =  a2 * (sigma2 / 2)**2 / ((x - mu2)**2 + (sigma2 / 2)**2)
    return lorentz2
    return gauss2

def cropData(x, y, xmin, xmax):
    """
    Funkce pro orezani dat zadanim minima a maxima v x
    """
    mask = (x > xmin) & (x < xmax)
    x_crop = x[mask]
    y_crop = y[mask]
    return x_crop, y_crop
    
def plot(x01, x02, p0, bounds, y_plot):
    #x1, y1 = cropData(x, y, x01-0.12, x02+0.12)
    x1, y1 = cropData(x, y, x01-0.5, x02+0.5)
    params = fit_gauss_two(x1, y1, p0=p0, bounds=bounds)
    print(*(map(lambda y: '{:.3f}'.format(y), params)))
    #fwhm = 2*np.sqrt(2*np.log(2))*params[1]   # Already implemented in functions
    fwhm = params[1] 
    print(f'FWHM of {x01} = {fwhm:.4f}')
    crystal_size = 0.94 * K_alfa_1 / np.radians(fwhm) / np.cos(np.radians(params[3]/2))
    print(f'crystal_size = {crystal_size:.2f} A')
    y_plot = y_plot + two_gaussians(x, *params)
    fig, ax = plt.subplots(figsize =(6.4,4))
    ax.plot(x, y, label = 'Measured Data')
    ax.plot(x, first_gaussian(x, params[0], params[1], params[3]), label="K$_{α1}$ Lorentzian")
    ax.plot(x, second_gaussian(x, params[2], params[1], params[4]), label="K$_{α2}$ Lorentzian")
    ax.plot(x, two_gaussians(x, *params), label="K$_{α1}$ + K$_{α2}$ Lorentzian")
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.set_xlabel("$2θ$ [°]")
    ax.set_ylabel("Intensity [a.u.]")
    ax.set_xlim(round_down_even(x01)-0.5, round_down_even(x01)+0.5)
    ax.set_ylim(0,p0[0])
    ax.tick_params(which = 'both', right=True, top=True)
    ax.legend()
    plt.savefig(f'plots/xrdplot{x01}.pdf')


def main():
    data = np.loadtxt('data/xrd_bb_01 copy.ras')
    global x
    x = data[:,0]
    global y
    y = data[:,1] - 5300
    #background = signal.savgol_filter(y, window_length=1000, polyorder=5)
    #p = np.polyfit(x, y, 1)
    #background = np.polyval(p, x)
    #y = y - background
    global K_alfa_1
    K_alfa_1 = 1.540601  # Angstrom
    global K_alfa_2
    K_alfa_2 = 1.544430  # Angstrom
    y_plot = np.array(np.zeros(len(y)))
    
    gigafig, gigaaxs = plt.subplots(2, 2)
    
    x01 = 43.14
    x02 = 43.25
    p0 = [50000, 0.02, 20000, x01, x02]
    #bounds = ([20000, 0, 10000, 0, 0], [np.inf, 0.1, np.inf, 100, 100])
    bounds = ([20000, 0, 1000, 0, 0], [np.inf, 0.2, np.inf, 100, 100])
    plot(x01, x02, p0, bounds, y_plot)
    
    x01 = 50.28
    x02 = 50.39
    p0 = [100000, 0.1, 50000, x01, x02]
    bounds = ([0, 0, 10000, 0, 0], [np.inf, 0.2, np.inf, 100, 100])
    plot(x01, x02, p0, bounds, y_plot)
    
    x01 = 74
    x02 = 74.21
    p0 = [40000, 0.1, 20000, x01, x02]
    bounds = ([0, 0, 0, 0, 0], [np.inf, 0.2, np.inf, 100, 100])
    plot(x01, x02, p0, bounds, y_plot)
    
    x01 = 89.80
    x02 = 90.08
    p0 = [25000, 0.06, 12000, x01, x02]
    #bounds = ([0, 0, 0, 0, 0], [np.inf, 0.2, np.inf, 100, 100])
    bounds = ([0, 0, 0, 0, 0], [np.inf, 0.3, np.inf, 100, 100])
    plot(x01, x02, p0, bounds, y_plot)
    
    x01 = 95.0
    x02 = 95.4
    y = y + 300
    p0 = [2000, 0.08, 1000, x01, x02]
    bounds = ([0, 0, 0, 0, 0], [np.inf, 0.2, np.inf, 100, 100])
    plot(x01, x02, p0, bounds, y_plot)
    y = y - 300
    
    fig, ax = plt.subplots()
    ax.plot(x, y, label = 'Measured Data')
    ax.plot(x, y_plot, label="K$_{α1}$ + K$_{α2}$ Lorentzian")
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.set_xlabel("$2θ$ [°]")
    ax.set_ylabel("Intensity [a.u.]")
    ax.tick_params(which = 'both', right=True, top=True)
    ax.legend()
    plt.savefig(f'plots/xrdplot2.pdf')
    
    fig, ax = plt.subplots()
    ax.plot(x, y, label = 'Measured Data')
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.set_xlabel("$2θ$ [°]")
    ax.set_ylabel("Intensity [a.u.]")
    ax.tick_params(which = 'both', right=True, top=True)
    
    peak_positions = [39, 43.14, 54, 73.98, 89.8, 95.0]
    peak_intensity = [1500, 45000, 95000, 38000, 21000, 2000]
    peak_labels = ['K$_β$','(111)', '(200)', '(220)', '(311)', '(222)']

    # Annotate the peaks in the plot
    for i in range(len(peak_positions)):
        xpeak = peak_positions[i]
        #y = spectrum_data[x] # Assuming spectrum_data is a dictionary or list
        y = peak_intensity[i]
        label = peak_labels[i]
        ax.annotate(label, xy=(xpeak, y), xytext=(xpeak, y), ha='center', va='bottom')
    ax.legend()
    plt.savefig(f'plots/xrdplot.pdf')
    #plt.show()
    
    pdfs = ['plots/xrdplot.pdf', 'plots/xrdplot2.pdf', 'plots/xrdplot43.14.pdf', 'plots/xrdplot50.28.pdf', 'plots/xrdplot73.98.pdf', 'plots/xrdplot89.8.pdf', 'plots/xrdplot95.0.pdf']
    output_pdf = 'plots/combined.pdf'
    merger = PdfMerger()
    for pdf in pdfs:
        merger.append(pdf)
    merger.write(output_pdf)
    merger.close()
    
    
if __name__ == '__main__':
    main()