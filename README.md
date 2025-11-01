# Probability Distributions, Bayesian Probability, and Gradient Descent Group 3 

## Overview
This group project focuses on deepening our understanding of **probability distributions**, **Bayesian probability**, and **gradient descent**.
It combines both manual calculations and using libratries to do the probability distribution, bayesian probability and gradient descent.
Through this assignment, we aim to apply mathematical concepts to practical coding problems, visualized results, and interpreted how learning algorithms adjust parameters to minimize error.

## Part 1: Probability Distributions

In this section, we implemented the **bivariate normal probability density function (PDF)** from scratch using only basic Python operations, without relying on external statistical libraries.

### Steps Completed

1. Selected a dataset containing two continuous variables from an online source which is the NBA dataset.

2. Calculated the probability density values for each data point using the bivariate normal distribution formula:

<img width="1700" height="188" alt="Screenshot 2025-10-31 151308" src="https://github.com/user-attachments/assets/1c4c78c2-7b09-43c1-8422-fe419fa1d884" />


3. Visualized the computed PDF using Matplotlib:

   * Contour plots to represent density levels.
   * 3D surface plots to visualize the overall distribution shape.

### Skills Demonstrated

* Manual computation of probability density values.
* Understanding of mean, standard deviation, and correlation in a multivariate context.
* Data visualization and interpretation using Matplotlib.


## Part 2: Bayesian Probability

In this section, we applied **Bayes’ Theorem** to a text classification task using the **IMDb Movie Reviews Dataset**.

### Objective

To determine how the presence of specific keywords influences the probability that a movie review is positive or negative.

### Steps Completed

1. Loaded the dataset using Pandas.

2. Selected 2–4 positive keywords and 2–4 negative keywords.

3. Chose to compute **P(Positive | keyword)** as our conditional probability.

4. Calculated the following for each keyword:

   * **Prior:** P(Positive)
   * **Likelihood:** P(keyword | Positive)
   * **Marginal:** P(keyword)
   * **Posterior:** P(Positive | keyword) using Bayes’ Theorem

5. Presented results in Markdown tables within the Jupyter Notebook.

### Skills Demonstrated

* Application of Bayes’ Theorem in text-based data analysis.
* Manual computation of probabilities using frequency counts.
* Clear understanding of conditional and marginal probabilities.

## Part 3: Manual Gradient Descent Calculation

In this section, we manually performed **three iterations** of the gradient descent algorithm for a simple linear regression model:

[
y = mx + b
]

### Given

* Initial values: ( m_0 ) and ( b_0 )
* Learning rate
* Data points: (x₁, y₁), (x₂, y₂), …

### Steps Completed

1. Computed predicted values

2. Derived the gradient of the Mean Squared Error (MSE) cost function with respect to both ( m ) and ( b ).

3. Updated ( m ) and ( b ) iteratively using

4. Repeated the process for three iterations, with each group member performing one update step.

### Observations

* Both ( m ) and ( b ) moved in a direction that reduced the error after each iteration.
* The rate of change slowed down, showing that gradient descent was converging toward the minimum.

### Skills Demonstrated

* Understanding of how gradients guide optimization.
* Manual derivation of cost function derivatives.
* Step-by-step mathematical computation of model parameter updates.

## Part 4: Gradient Descent in Code

In this section, we implemented the gradient descent algorithm in Python using **SciPy** to automate the process performed manually in Part 3.

### Steps Completed

1. Defined the cost function and its gradient.
2. Updated the parameters ( m ) and ( b ) iteratively using the gradient descent update rule.
3. Computed predictions for each iteration.
4. Visualized the following using Matplotlib:

   * Error vs. Iterations
   * Changes in ( m ) and ( b ) over time

### Skills Demonstrated

* Translating mathematical operations into code.
* Using SciPy for numerical optimization.
* Data visualization and interpretation of patterns.

## Group Members and Contributions
Here is the document showing every memmbers contributions: https://docs.google.com/document/d/1_6sNshdg9O6prrsWg9UJjJK2mUNr01STXJy0A2fjkUQ/edit?usp=sharing 

## How to Run the Code

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-repo-name>.git
   cd <your-repo-name>
   ```
2. Install required dependencies:

   ```bash
   pip install numpy pandas matplotlib scipy
   ```
3. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```
4. Open each notebook in order:

   * `part1_probability_distributions.ipynb`
   * `part2_bayesian_probability.ipynb`
   * `part4_gradient_descent_code.ipynb`

## Files Included

| File Name                               | Description                                                   |
| --------------------------------------- | ------------------------------------------------------------- |
| `part1_probability_distributions.ipynb` | Implementation and visualization of the bivariate PDF         |
| `part2_bayesian_probability.ipynb`      | Bayes’ Theorem implementation using IMDb dataset              |
| `part3_manual_gradient_descent.pdf`     | Manual calculations and handwritten work for gradient descent |
| `part4_gradient_descent_code.ipynb`     | Gradient descent implementation using SciPy                   |
| `group_contributions.pdf`               | Proof of contribution from all members                        |


## Results Summary

* The bivariate PDF visualization showed how data density changes with variable correlation using the NBA data set.
* The Bayesian probability results confirmed the impact of certain keywords on sentiment classification.
* Manual gradient descent demonstrated step-by-step convergence behavior.
* The coded version verified the same results with automated calculations and visual plots.
