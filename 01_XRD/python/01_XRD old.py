import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
from functools import partial
from matplotlib.ticker import AutoMinorLocator 

def two_gaussians(x, a1, sigma1, a2, mu1=0, mu2=0):
    gauss1 = a1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2))
    gauss2 = a2 * np.exp(-(x - mu2)**2 / (2 * sigma1**2))
    return gauss1 + gauss2

def fit_gauss_two(x, y, bounds, p0, mu1, mu2):
    fixed_gaussians = partial(two_gaussians, mu1=mu1, mu2=mu2)
    popt, pcov = curve_fit(fixed_gaussians, x, y, p0=p0, bounds=bounds, ftol=1e-10)
    return popt

def first_gaussian(x, a1, sigma1, mu1=0):
    gauss1 = a1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2))
    return gauss1

def second_gaussian(x, a2, sigma2, mu2=0):
    gauss2 = a2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2))
    return gauss2

def cropData(x, y, xmin, xmax):
    """
    Funkce pro orezani dat zadanim minima a maxima v x
    """
    mask = (x > xmin) & (x < xmax)
    x_crop = x[mask]
    y_crop = y[mask]
    return x_crop, y_crop
    

def main():
    data = np.loadtxt('data/xrd_bb_01 copy.ras')
    x = data[:,0]
    y = data[:,1] - 5300
    
    y_plot = np.array(np.zeros(len(y)))
    
    x01 = 43.14
    x02 = 43.25
    p0 = [50000, 0.1, 10000]
    x1, y1 = cropData(x, y, x01-0.5, x01+0.5)
    bounds = ([0, 0, 0], [np.inf, 0.2, np.inf])
    params = fit_gauss_two(x1, y1, p0=p0, bounds=bounds, mu1=x01, mu2=x02)
    print(*(map(lambda y: '{:.2f}'.format(y), params)))
    fwhm = 2*np.sqrt(2*np.log(2))*params[1]
    print(f'FWHM of {x01} = {fwhm:.4f}')
    fixed_gaussians = partial(two_gaussians, mu1=x01, mu2=x02)
    y_plot = y_plot + fixed_gaussians(x, *params)
    fig, ax = plt.subplots()
    ax.plot(x, y, label = 'Measured Data')
    ax.plot(x, fixed_gaussians(x, *params), label="Double Gaussian")
    ax.plot(x, first_gaussian(x, params[0], params[1], mu1=x01), label="K$_{α1}$ Gaussian")
    ax.plot(x, second_gaussian(x, params[2], params[1], mu2=x02), label="K$_{α2}$ Gaussian")
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.set_xlim(x01-0.5, x01+0.5)
    ax.set_ylim(0,p0[0])
    
    ax.legend()
    plt.savefig(f'plots/xrdplot{x01}.pdf')
    
    x01 = 50.28
    x02 = 50.39
    p0 = [100000, 0.1, 20000]
    bounds = ([0, 0, 0], [np.inf, 0.2, np.inf])
    params = fit_gauss_two(x, y, p0=p0, bounds=bounds, mu1=x01, mu2=x02)
    print(*(map(lambda y: '{:.2f}'.format(y), params)))
    fwhm = 2*np.sqrt(2*np.log(2))*params[1]
    print(f'FWHM of {x01} = {fwhm:.4f}')
    fixed_gaussians = partial(two_gaussians, mu1=x01, mu2=x02)
    y_plot = y_plot + fixed_gaussians(x, *params)
    fig, ax = plt.subplots()
    ax.plot(x, y, label = 'Measured Data')
    ax.plot(x, fixed_gaussians(x, *params), label="Double Gaussian")
    ax.plot(x, first_gaussian(x, params[0], params[1], mu1=x01), label="K$_{α1}$ Gaussian")
    ax.plot(x, second_gaussian(x, params[2], params[1], mu2=x02), label="K$_{α2}$ Gaussian")
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.set_xlim(x01-0.5, x01+0.5)
    ax.set_ylim(0,p0[0])
    ax.legend()
    plt.savefig(f'plots/xrdplot{x01}.pdf')
    
    x01 = 73.98
    x02 = 74.19
    p0 = [40000, 0.1, 20000]
    bounds = ([0, 0, 0], [np.inf, 0.2, np.inf])
    params = fit_gauss_two(x, y, p0=p0, bounds=bounds, mu1=x01, mu2=x02)
    print(*(map(lambda y: '{:.2f}'.format(y), params)))
    fwhm = 2*np.sqrt(2*np.log(2))*params[1]
    print(f'FWHM of {x01} = {fwhm:.4f}')
    fixed_gaussians = partial(two_gaussians, mu1=x01, mu2=x02)
    y_plot = y_plot + fixed_gaussians(x, *params)
    fig, ax = plt.subplots()
    ax.plot(x, y, label = 'Measured Data')
    ax.plot(x, fixed_gaussians(x, *params), label="Double Gaussian")
    ax.plot(x, first_gaussian(x, params[0], params[1], mu1=x01), label="K$_{α1}$ Gaussian")
    ax.plot(x, second_gaussian(x, params[2], params[1], mu2=x02), label="K$_{α2}$ Gaussian")
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.set_xlim(x01-0.5, x01+0.5)
    ax.set_ylim(0,p0[0])
    ax.legend()
    plt.savefig(f'plots/xrdplot{x01}.pdf')
    
    x01 = 89.80
    x02 = 90.08
    p0 = [25000, 0.08, 12000]
    bounds = ([0, 0, 0], [np.inf, 0.11, np.inf])
    params = fit_gauss_two(x, y, p0=p0, bounds=bounds, mu1=x01, mu2=x02)
    print(*(map(lambda y: '{:.2f}'.format(y), params)))
    fwhm = 2*np.sqrt(2*np.log(2))*params[1]
    print(f'FWHM of {x01} = {fwhm:.4f}')
    fixed_gaussians = partial(two_gaussians, mu1=x01, mu2=x02)
    y_plot = y_plot + fixed_gaussians(x, *params)
    fig, ax = plt.subplots()
    ax.plot(x, y, label = 'Measured Data')
    ax.plot(x, fixed_gaussians(x, *params), label="Double Gaussian")
    ax.plot(x, first_gaussian(x, params[0], params[1], mu1=x01), label="K$_{α1}$ Gaussian")
    ax.plot(x, second_gaussian(x, params[2], params[1], mu2=x02), label="K$_{α2}$ Gaussian")
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.set_xlim(x01-0.5, x01+0.5)
    ax.set_ylim(0,p0[0])
    ax.legend()
    plt.savefig(f'plots/xrdplot{x01}.pdf')
    
    fig, ax = plt.subplots()
    ax.plot(x, y, label = 'Measured Data')
    ax.plot(x, y_plot, label="Double Gaussian")
    ax.legend()
    plt.savefig(f'plots/xrdplot2.pdf')
    
    fig, ax = plt.subplots()
    ax.plot(x, y, label = 'Measured Data')
    ax.legend()
    plt.savefig(f'plots/xrdplot.pdf')
    #plt.show()
    
    
if __name__ == '__main__':
    main()