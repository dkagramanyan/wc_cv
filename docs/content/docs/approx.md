---
title: "Approximation"
weight: 16
---

## gaussian_fit(x, y, mu=1, sigma=1, amp=1)

Аппроксимация заданных точек нормальной функцией

**Параметры:**
- `x`: list (n,)
- `y`: list (n,)
- `mu`: float
- `sigma`: float
- `amp`: float

**Возвращает:** mus, sigmas, amps

## gaussian_fit_bimodal(x, y, mu1=100, mu2=240, sigma1=30, sigma2=30, amp1=1, amp2=1)

Аппроксимация заданных точек бимодальной нормальной функцией

**Параметры:**
- `x`: list (n,)
- `y`: list (n,)
- `mu1`: float
- `mu2`: float
- `sigma1`: float
- `sigma2`: float
- `amp1`: float
- `amp2`: float

**Возвращает:** mus, sigmas, amps

## gaussian_fit_termodal(x, y, mu1=10, mu2=100, mu3=240, sigma1=10, sigma2=30, sigma3=30, amp1=1, amp2=1, amp3=1)

Аппроксимация заданных точек термодальной нормальной функцией

**Параметры:**
- `x`: list (n,)
- `y`: list (n,)
- `mu1`: float
- `mu2`: float
- `mu3`: float
- `sigma1`: float
- `sigma2`: float
- `sigma3`: float
- `amp1`: float
- `amp2`: float
- `amp3`: float

**Возвращает:** mus, sigmas, amps

## lin_regr_approx(x, y)

Аппроксимация распределения линейной функцией и создание графика по параметрам распределения

**Параметры:**
- `x`: list (n,)
- `y`: list (n,)

**Возвращает:** (x_pred, y_pred), k, b, angle, score

## bimodal_gauss_approx(x, y)

Аппроксимация распределения бимодальным гауссом

**Параметры:**
- `x`: list (n,)
- `y`: list (n,)

**Возвращает:** (x_gauss, y_gauss), mus, sigmas, amps

