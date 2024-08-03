# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 12:33:16 2024

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import romberg
from scipy.interpolate import interp1d
from numba import jit
import time


# define timer for calculating time consumption of a function, will be a decorator 
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function {func.__name__} took {execution_time} seconds to execute.")
        return result
    return wrapper


def fForRomb(p, e0, dosa):
    interpolator = interp1d(e0, dosa, kind='cubic', fill_value="extrapolate")
    return interpolator(p)


def rombergInt(a, b, func, e0, dosa):
    result = romberg(func, a, b, args=(e0, dosa), divmax=10)
    return result


def monteCarloInt(a, b, func, e0, dosa, num_samples=100000):
    samples = np.random.uniform(a, b, num_samples)
    values = func(samples, e0, dosa)
    integral = (b - a) * np.mean(values)
    return integral


@timer
def Green_1Totdos():
    el = -2
    eu = 2
    Ne = 100
    es = (eu - el) / (Ne - 1)
    gam = 1/2
    zim = complex(0., 1.0)
    delta = 1.e-4
    
    e0 = np.linspace(el, eu, Ne)
    dosa = np.zeros(Ne)

    for i in range(Ne):
        g00 = e0[i] / (np.sqrt((e0[i] + zim * delta)**2 - 4 * gam**2) * abs(e0[i]))
        dosa[i] = -np.imag(g00) / np.pi

    x = 0.1
    eT = np.arange(el, eu + x, x)
    intdos = np.zeros(len(eT))

    for nt in range(len(eT)):
        '''
        One can select using Romberg integration or Monte Carlo integration
        
        '''
        # Romberg integration
        #intdos[nt] = rombergInt(el, eT[nt], fForRomb, e0, dosa)
        
        # Monte Carlo integration
        intdos[nt] = monteCarloInt(el, eT[nt], fForRomb, e0, dosa)
        print(f"E0={eT[nt]:9.4f}, integrated dos={intdos[nt]:14.6e}")
    

    plt.plot(e0, dosa, 'k', label='D(E)')
    plt.scatter(e0,dosa, label='D(E), numerical')
    plt.plot(eT, 2 * intdos, 'k:', linewidth=2, label='2*N(E)')
    plt.legend(loc='best')
    plt.xlabel('E (Ha)')
    plt.ylabel('D(E) (1/Ha) and 2*N(E)')
    plt.title('Density of States and 2*Total Density of States')
    plt.grid()
    plt.show()

Green_1Totdos()

#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 原始数据
e0 = np.linspace(-2, 2, 1000)
gam = 0.5
delta = 1e-4
zim = 1j
dosa = np.array([-np.imag(e / (np.sqrt((e + zim * delta) ** 2 - 4 * gam ** 2) * abs(e))) / np.pi for e in e0])

# 定义插值函数
def fForRomb(p, e0, dosa):
    interpolator = interp1d(e0, dosa, kind='cubic', fill_value="extrapolate")
    return interpolator(p)

# 生成一组点并计算插值函数在这些点上的值
p_values = np.linspace(-2, 2, 1000)
interp_values = fForRomb(p_values, e0, dosa)

# 绘制插值函数的曲线表示
plt.plot(p_values, interp_values, label='Interpolated Function')
plt.xlabel('Energy (Ha)')
plt.ylabel('Density of States (1/Ha)')
plt.title('Interpolated Function Plot')
plt.legend()
plt.grid(True)
plt.show()

#%%

