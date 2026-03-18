# Gaussian Naive Bayes: A Comprehensive Guide

Gaussian Naive Bayes is a probabilistic classifier based on two main pillars: **Bayesian statistics** and the assumption of a **Normal (Gaussian) distribution** for continuous data.

---

## 1. The "Bayes" Part (Bayes' Theorem)

At its core, the model uses **Bayes' Theorem** to calculate the probability of a sample belonging to a class $C$ (e.g., "Spam" or "Not Spam") given its observed features $x$:

$$P(C|x) = \frac{P(x|C) \cdot P(C)}{P(x)}$$

### Breakdown of Terms:
* **$x$**: The input data point, usually a vector of multiple features (columns).
* **$P(C|x)$ (Posterior)**: The probability we want to find (e.g., "What is the chance this is Spam given these specific words?").
* **$P(C)$ (Prior)**: The baseline probability of a class based on training data (e.g., "In my training set, 50% of all emails are Spam").
* **$P(x|C)$ (Likelihood)**: The probability of seeing these specific features if we already know the class is $C$.
* **$P(x)$ (Evidence)**: The total probability of seeing these features. Since this is the same for all classes during a single prediction, we ignore it and just compare the numerators.

---

## 2. The "Naive" Part (Independence Assumption)

In real life, features often depend on each other. For example, if a patient has a "High Fever," they are also likely to have a "High Heart Rate."

The model is **"Naive"** because it ignores these relationships. It assumes that every feature is **conditionally independent** of every other feature. This allows us to simplify the massive probability $P(x|C)$ into a simple multiplication of individual probabilities:

$$P(x_1, x_2, \dots, x_n | C) = P(x_1|C) \cdot P(x_2|C) \cdot \dots \cdot P(x_n|C)$$

---

## 3. The "Gaussian" Part (Handling Continuous Numbers)

If your features are categories (like "Color"), you can just count them. But if your features are continuous numbers (like "Weight" or "Temperature"), you can't "count" how many times $72.456$°C appears.

Instead, we assume the data for each class follows a **Gaussian (Normal) Distribution**—the classic bell curve.



### During Training (`fit`)
For every class, the model calculates the **Mean** ($\mu$) and **Variance** ($\sigma^2$) of each feature. These two numbers define the shape of the bell curve for that specific feature in that class.

### During Prediction (`predict`)
The model plugs the new value into the **Gaussian Probability Density Function (PDF)**:

$$f(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

---

## Technical Details

### What `var_smoothing` actually does
In Gaussian Naive Bayes, we divide by the variance ($\sigma^2$) when calculating the probability density. If the variance is zero (or extremely close to it), the math breaks down—you either get a "Division by Zero" error or the probability shoots up to infinity, which crashes the program.

`var_smoothing` takes a tiny portion of the **largest variance** in your entire dataset and adds it to every feature's variance. This acts as a stability buffer.

### Why use a Normal Distribution?
**The Central Limit Theorem:** In nature and social science, if you collect enough data points for a measurement (like height, blood pressure, or test scores), they almost always naturally form a bell curve.