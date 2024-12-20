# Quantile Regression Analysis Project

This project implements quantile regression analysis to explore the relationship between predictors and a response variable, specifically focusing on delivery time based on the distance to the customer. It provides various statistical tests, visualizations, and comparisons between linear and quantile regression models. The goal is to understand how different quantiles of the response variable can be modeled and how they differ from traditional linear regression.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contributing](#contributing)
- [Contact Information](#contact-information)

## Features

- Implementation of quantile regression using the `statsmodels` library.
- Visualization of quantile regression results with smoothed lines.
- Statistical tests including Wald Test, Score Test, and Lagrange Multiplier Test.
- Comparison of quantile coefficients across different quantiles.
- Support for weighted quantile regression and associated tests.
- Enhanced plotting functions to compare linear and quantile regression models.
- Calculation of quantile loss for model evaluation.

## Technologies Used

- Python
- NumPy
- Matplotlib
- Scikit-learn
- Statsmodels
- SciPy
- Jupiter Notebook

## Getting Started

To get started with this project, follow these steps:

1. **Clone the repository** to your local machine:

    ```bash
    git clone 
    ```

2. Navigate to the project directory:

    ```bash
    cd quantileRegression
    ```

3. Install the required dependencies using pip:

    ```bash
    pip install notebook numpy matplotlib scikit-learn statsmodels scipy
    ```

## Usage

This project includes four key files:

1. `functions.py`: This Python script contains all the functions necessary for performing quantile regression analysis, including visualization functions, statistical tests (Wald, Score, and Lagrange Multiplier tests), and loss functions. You can import this module in your Jupyter notebooks to access these functions.

2. `quantileRegression.ipynb`: This Jupyter Notebook is used for performing standard quantile regression analysis. It includes steps for loading the dataset, fitting quantile regression models, visualizing the results, and conducting statistical tests to evaluate the models.

3. `quantileRegressionWeighted.ipynb`: This notebook extends the analysis to weighted quantile regression. It allows you to explore how incorporating weights affects the regression results and includes visualizations and statistical tests specific to weighted models.

4. `quantileRegression_scenarios.ipynb`: This notebook is designed to analyze different scenarios using quantile regression. It includes comparisons of linear regression and quantile regression under various conditions, providing insights into how the models perform across different quantiles and datasets.

To run the notebooks, open them in Jupyter Notebook or Jupyter Lab, and execute the cells sequentially to see the results.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.

2. Create a new branch:

   ```bash
   git checkout -b feature/feature-name
   ```

3. Commit your changes:

   ```bash
   git commit -m "Add feature description"
   ```

4. Push to the branch:

   ```bash
   git push origin feature/feature-name
   ```

5. Open a pull request.

## Contact Information

For questions or support, please contact [Me](mailto:chouaiba629@gmail.com).
