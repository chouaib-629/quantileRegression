import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from scipy import stats

# Visualization Function
def plot_quantile_regression(X, y, results_list, taux):
    """
    Function to display quantile regression results with smoothed lines.
    Handles multi-predictor models by fixing other predictors at their mean.
    """
    plt.figure(figsize=(10, 6))
    
    # Scatter plot of observed data
    plt.scatter(X[:, 1], y, alpha=0.5, label='Observed Data', color='blue')
    
    # Create a grid for x1 (varying x1 while keeping x2, x3 fixed)
    x1_grid = np.linspace(X[:, 1].min(), X[:, 1].max(), 500)
    x2_fixed = X[:, 2].mean()  # Fix x2 at its mean
    x3_fixed = X[:, 3].mean()  # Fix x3 at its mean
    
    # Construct grid for all predictors
    X_grid = np.column_stack([np.ones_like(x1_grid), x1_grid, np.full_like(x1_grid, x2_fixed), np.full_like(x1_grid, x3_fixed)])
    
    # Plot the quantile regression lines
    for result, taux in zip(results_list, taux):
        y_pred = result.predict(X_grid)  # Predict on the grid
        plt.plot(x1_grid, y_pred, label=f'Quantile τ = {taux}', linewidth=2)
    
    plt.xlabel('Distance to Customer (km)')
    plt.ylabel('Delivery Time (Minutes)')
    plt.title('Quantile Regressions for Delivery Time')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def quantile_loss(y_true, y_pred, quantile):
    """
    Quantile loss function for evaluating quantile regression
    """
    error = y_true - y_pred
    loss = np.maximum(quantile * error, (quantile - 1) * error).mean()
    return loss

# Wald Test
def wald_test(result):
    """
    Perform the Wald test to check if the coefficients are significantly different from zero.
    """
    # Extract the coefficient and its standard error for the first predictor
    coef = result.params[1]  # Coefficient of 'distance_to_customer'
    se = result.bse[1]       # Standard error of the coefficient

    # Calculate Wald statistic and p-value
    wald_stat = coef / se
    p_value = 2 * (1 - stats.norm.cdf(abs(wald_stat)))  # Two-tailed test

    print("\nWald Test Results:")
    print(f"  Coefficient: {coef:.3f}")
    print(f"  Wald Statistic: {wald_stat:.3f}")
    print(f"  P-value: {p_value:.3f}")

# Score Test (Lagrange Multiplier Test)  
def score_test(result, X, y, tau):
    """
    Perform the Score Test (Lagrange Multiplier Test) for beta = 0.
    """
    # Compute residuals
    residuals = y - np.dot(X, result.params)

    # Score statistic
    score_stat = np.sum(residuals * X[:, 1]) / np.sum(X[:, 1] ** 2)

    # Compute p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(score_stat)))  # Two-tailed test

    print("\nScore Test Results:")
    print(f"  Score Statistic: {score_stat:.3f}")
    print(f"  P-value: {p_value:.3f}")

# Lagrange Multiplier (LM) Test 
def lm_test(result, X, y):
    """
    Perform the LM test based on the difference in log-likelihood.
    """
    # Number of observations
    n = len(y)

    # Residuals
    residuals = y - np.dot(X, result.params)

    # LM Statistic
    lm_stat = np.sum((residuals * 2) * (X[:, 1] * 2)) / n

    # Compute p-value using chi-squared distribution
    p_value = 1 - stats.chi2.cdf(lm_stat, df=1)

    print("\nLagrange Multiplier (LM) Test Results:")
    print(f"  LM Statistic: {lm_stat:.3f}")
    print(f"  P-value: {p_value:.3f}")
    
# Compare Quantile Coefficients
def compare_quantile_coeffs(X, y, taux1, taux2):
    """
    Compare the coefficients of two quantile regressions.
    """
    # Fit quantile regression for two quantiles
    model1 = QuantReg(y, X).fit(q=taux1)
    model2 = QuantReg(y, X).fit(q=taux2)

    # Coefficients and variances
    coef1 = model1.params[1]
    coef2 = model2.params[1]
    var1 = model1.cov_params()[1, 1]
    var2 = model2.cov_params()[1, 1]

    # Calculate difference and t-statistic
    diff = coef1 - coef2
    se_diff = np.sqrt(var1 + var2)
    t_stat = diff / se_diff

    # Compute p-value
    p_value_diff = 2 * (1 - stats.norm.cdf(abs(t_stat)))  # Two-tailed test

    print(f"\nComparison of Coefficients between τ = {taux1} and τ = {taux2}:")
    print(f"  Difference of Coefficients: {diff:.3f}")
    print(f"  T-statistic: {t_stat:.3f}")
    print(f"  P-value: {p_value_diff:.3f}") 

# Visualization Function Weighted
def plot_quantile_regression_weighted(X, y, weights, results_list, taux):
    """
    Function to display quantile regression results with weights.
    Handles multi-predictor models by fixing other predictors at their mean.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 1], y, alpha=0.5, label='Observed Data', color='blue', s=weights * 100)
    
    x1_grid = np.linspace(X[:, 1].min(), X[:, 1].max(), 500)
    x2_fixed = X[:, 2].mean()
    x3_fixed = X[:, 3].mean()
    X_grid = np.column_stack([np.ones_like(x1_grid), x1_grid, np.full_like(x1_grid, x2_fixed), np.full_like(x1_grid, x3_fixed)])
    
    for result, taux in zip(results_list, taux):
        y_pred = result.predict(X_grid)
        plt.plot(x1_grid, y_pred, label=f'Quantile τ = {taux}', linewidth=2)
    
    plt.xlabel('Distance to Customer (km)')
    plt.ylabel('Delivery Time (minutes)')
    plt.title('Quantile Regression Results with Weights')
    plt.legend()
    plt.grid(True)
    plt.show()
  
# Weighted Quantile Loss
def quantile_loss_weighted(y_true, y_pred, weights, quantile):
    """
    Weighted quantile loss function for evaluating quantile regression
    """
    error = y_true - y_pred
    loss = np.average(
        np.maximum(quantile * error, (quantile - 1) * error),
        weights=weights
    )
    return loss

# Weighted Wald Test
def wald_test_weighted(result):
    """
    Perform the Wald test to check if the coefficients are significantly different from zero for weighted quantile regression.
    """
    coef = result.params[1]
    se = result.bse[1]
    wald_stat = coef / se
    p_value = 2 * (1 - stats.norm.cdf(abs(wald_stat)))
    print("\nWald Test Results (Weighted):")
    print(f"  Coefficient: {coef:.3f}")
    print(f"  Wald Statistic: {wald_stat:.3f}")
    print(f"  P-value: {p_value:.3f}")

# Weighted Lagrange Multiplier (LM) Test   
def lm_test_weighted(result, X, y, weights):
    """
    Perform the LM test based on the difference in log-likelihood for weighted quantile regression.
    """
    residuals = (y - np.dot(X, result.params)) * weights
    lm_stat = np.sum((residuals * 2) * (X[:, 1] * 2)) / len(y)
    p_value = 1 - stats.chi2.cdf(lm_stat, df=1)
    print("\nLM Test Results (Weighted):")
    print(f"  LM Statistic: {lm_stat:.3f}")
    print(f"  P-value: {p_value:.3f}")
    
# Enhanced plotting and comparison function
def plot_and_compare_errors(X, y, quantiles, scenario_title):
    """
    Function to compare linear regression and quantile regression with visual enhancements.
    """
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Add constant for intercept
    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)

    # Fit linear regression
    linear_model = sm.OLS(y_train, X_train_const).fit()
    y_pred_linear = linear_model.predict(X_test_const)

    # Fit quantile regressions for each quantile
    quantile_models = [QuantReg(y_train, X_train_const).fit(q=q) for q in quantiles]
    y_preds_quantile = [model.predict(X_test_const) for model in quantile_models]

    # Calculate errors
    mae_linear = mean_absolute_error(y_test, y_pred_linear)
    mae_quantiles = [mean_absolute_error(y_test, y_pred) for y_pred in y_preds_quantile]

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.5, label='Observed Data', color='blue')

    # Plot linear regression line
    plt.plot(X_test, y_pred_linear, label=f'Linear Regression (MAE = {mae_linear:.2f})', color='red', linewidth=2)

    # Plot quantile regression lines
    for q, y_pred, mae in zip(quantiles, y_preds_quantile, mae_quantiles):
        plt.plot(X_test, y_pred, label=f'Quantile Regression (τ = {q}, MAE = {mae:.2f})', linewidth=2)

    # Annotate outliers or regions of high variability
    plt.scatter(X[np.abs(y - np.mean(y)) > 1.5 * np.std(y)], 
                y[np.abs(y - np.mean(y)) > 1.5 * np.std(y)], 
                color='orange', label='Potential Outliers', s=50, edgecolor='black')

    # Add labels and legend
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(scenario_title)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Residuals plot
    residuals_linear = y_test - y_pred_linear
    residuals_quantiles = [y_test - y_pred for y_pred in y_preds_quantile]
    plt.figure(figsize=(10, 6))
    plt.hist(residuals_linear, bins=15, alpha=0.5, label='Linear Regression Residuals', color='red')
    for q, residuals in zip(quantiles, residuals_quantiles):
        plt.hist(residuals, bins=15, alpha=0.5, label=f'Quantile τ = {q} Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print the error comparison
    print(f"{scenario_title}")
    print(f"Linear Regression MAE: {mae_linear:.2f}")
    for q, mae in zip(quantiles, mae_quantiles):
        print(f"Quantile Regression (τ = {q}) MAE: {mae:.2f}")
    print("-" * 50)