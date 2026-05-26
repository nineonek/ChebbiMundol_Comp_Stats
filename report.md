# Predicting House Sale Prices with Statistical Modeling and Neural Networks
## Project Report

**Authors:**
* Student 1: Chebbi Redhouan | 24-508-814
* Student 2: Mundol Salih Ishaan | 24-508-798

**Date:** Computational Statistics, Spring 2026, May

### Abstract
Who would've known that predicting house prices was like solving a mystery novel? Spoiler alert: this mystery involves a lot of numbers, some plots, graphs, and the realization that *location*, *size*, and *quality* are basically the main trifecta of real estate valuation.

We started by exploring the data and figuring out that `SalePrice` is heavily right-skewed (skewness = 1.88), so we log-transformed it to make it usable for parametric tests. From there we used one-way ANOVA to identify 7 features that significantly impact price, then studied how these features interact together using two-way ANOVA and a $2^3$ factorial design. We found that `OverallQual` and `KitchenQual` interact significantly — a good kitchen matters more in a high-quality home.

We then built two predictive models. A parametric OLS regression with 9 features reached $R^2 = 0.832$, with `GrLivArea` clearly dominating after log-transforming it. A neural network (MLP) went further by engineering new features and capturing non-linear patterns, achieving a training RMSE under 0.07 on the log scale.

## 1. Introduction

### Objective
The primary goal of this project is to predict house sale prices using the *House Prices — Advanced Regression Techniques* dataset from Kaggle. However, we're not just throwing our data at an algorithm and hoping for the best (we save that for real estate agents...). Instead, we (obviously) apply rigorous statistical methods to:
* **Understand the data** through classical statistical inference
* **Identify significant features** that genuinely influence price
* **Analyze feature interactions** using factorial design
* **Build predictive models** with parametric and non-parametric techniques
* **Validate and interpret** our results

### Methodology Overview
Our approach follows a systematic pipeline:
1. **Data Exploration & Transformation** — Make the data "normal"
2. **Feature Significance Testing** — ANOVA to filter out the statistical noise
3. **Interaction Analysis** — $2^3$ Factorial design to understand when features matter *together*
4. **Parametric Modeling** — Linear regression with OLS, Ridge, and Lasso
5. **Non-Parametric Modeling** — Neural networks to capture complex relationships

### Key Findings Preview
* The untransformed `SalePrice` is aggressively right-skewed (skewness = 1.88)
* Log transformation brings it to an approximately normal distribution (skewness = 0.12)
* **7 features** passed the significance test ($p < 0.05$) in one-way ANOVA
* Quite a few significant interactions found — example: `OverallQual` $	imes$ `KitchenQual` ($p = 10^{-10}$)
* `GrLivArea` dominates the OLS model with the highest F-statistic ($F = 624$)

## 2. Data and Preprocessing

### 2.1 Data Source
The dataset we used is the *House Prices — Advanced Regression Techniques* available on Kaggle. It contains information on 1460 houses sold in Ames, Iowa, with 79 features describing pretty much every aspect of a residential house — from the living area to kitchen quality to roof type. The target variable is `SalePrice`, the sale price of each house.

`train.csv` has 1460 observations with prices, and `test.csv` has 1459 observations without prices — that's the file we use to generate the Kaggle submission at the end.

### 2.2 Preprocessing Decisions

#### Transformation
**Problem:** The original `SalePrice` distribution looks like an angry hockey stick—skewed to the right with outliers that make you question whether someone really paid $755,000 for a house in Iowa.

Key statistics before transformation:
* Mean: $180,921
* Variance: $4.73 \times 10^9$ (that's a lot of scattered houses!)
* Skewness: **1.88** (extremely right-skewed)

**Solution:** Natural logarithm transformation using $\log(1 + \text{SalePrice})$.

Why this works:
* Converts multiplicative relationships to additive ones
* Stabilizes variance across price ranges (homoscedasticity)
* Reduces the influence of outliers

After log transformation:
* Mean: 12.021
* Skewness: **0.12** (basically normal now!)
* The distribution went from "angry hockey stick" to "nice bell curve" ✓

#### Normality Validation
Using the **Shapiro-Wilk test** on the log-transformed price:
* Test Statistic: 0.990 (very close to 1.0 ✓)
* $p$-value: $< 0.001$

**Interpretation:** Technically, we reject normality at $\alpha = 0.05$, but with $n = 1,460$ observations, Shapiro-Wilk becomes *suspiciously* strict. The Q-Q plot confirms the data hugs the normal line beautifully except at the extremes.

**Conclusion:** The transformation is sufficiently normal for parametric tests like $t$-tests and ANOVA. ✓

## 3. Classical Statistical Inference

### 3.1 Descriptive Statistics
Before doing anything we looked at the basic statistics of `SalePrice` and a few key features to understand what we're working with.

For `SalePrice` we get:
* Mean: $180,921
* Variance: $6.31 \times 10^9$
* Std: $79,442
* Skewness: $1.883$ — heavily right-skewed

The distribution is clearly not normal — there's a strong right tail with a few very expensive houses pulling the mean up. That's why we applied a log-transformation (see section 2.2), which brings the skewness down to $0.121$ and the variance to $0.160$, way more reasonable.

We also looked at 4 key numerical features:

| | `GrLivArea` | `TotalBsmtSF` | `OverallQual` | `YearBuilt` |
| :--- | :--- | :--- | :--- | :--- |
| **mean** | 1515.46 | 1057.43 | 6.10 | 1971.27 |
| **std** | 525.48 | 438.71 | 1.38 | 30.20 |
| **var** | 276129.63 | 192462.36 | 1.91 | 912.22 |
| **skew** | 1.37 | 1.52 | 0.22 | -0.61 |

What's interesting is that `GrLivArea` and `TotalBsmtSF` are also pretty skewed (1.37 and 1.52), meaning there are a few really large houses that pull away from the mean. `OverallQual` on the other hand is almost symmetric (skew = 0.22) with a mean around 6, which corresponds to "above average" quality for a typical house. `YearBuilt` is slightly left-skewed (skew = -0.61), meaning there are more recent houses than very old ones in the dataset.

### 3.2 Confidence Interval
Using the **$t$-distribution** (because we do not know the underlying variance):

**95% Confidence Interval:**
* Log scale: $[12.0003, 12.0421]$
* Original $: $[$180,434 ; $181,410]

**Interpretation:** We are 95% confident that the true mean house price for our localized sample space falls somewhere in this range. ✓

### 3.3 Hypothesis Test
**Hypotheses:**
$$H_0 : \mu = \$180,000$$
$$H_1 : \mu \neq \$180,000$$

**$t$-test Result:**
* $t$-statistic: -0.004
* $p$-value: 0.997

**Conclusion:** We fail to reject $H_0$. The mean price is NOT significantly different from $180,000. In other words, our sample data is basically *screaming* that $180,000 is the true average.

## 4. ANOVA

### 4.1 Features Analyzed
To find significant features, we went through 10 of them in order to analyze their impact on the final (transformed) Salesprice. These are:

| # | Feature | Levels | Description |
| :--- | :--- | :--- | :--- |
| 1 | `OverallQual` | 1–10 | Overall material and finish quality |
| 2 | `ExterQual` | Po, Fa, TA, Gd, Ex | Exterior material quality |
| 3 | `BsmtQual` | None, Po, Fa, TA, Gd, Ex | Basement height quality |
| 4 | `KitchenQual` | Po, Fa, TA, Gd, Ex | Kitchen quality |
| 5 | `FireplaceQu` | None, Po, Fa, TA, Gd, Ex | Fireplace quality |
| 6 | `CentralAir` | N, Y | Central air conditioning |
| 7 | `LotShape` | IR3, IR2, IR1, Reg | General shape of property |
| 8 | `LandSlope` | Sev, Mod, Gtl | Slope of property |
| 9 | `MoSold` | 1–12 | Month sold |
| 10 | `YrSold` | 2006–2010 | Year sold |

In order to measure and single out these important features and interactions, we will separate our analysis in two axes : One and Two-Way ANOVA.

### 4.2 One-Way ANOVA
One-way ANOVA was used to compare the mean log-SalePrice across the levels of each feature. The objective here was to test the null hypothesis ($H_0$) that the mean housing prices are identical across all levels of a given feature. For example: Would a property sold in 2006 be (on average) cheaper than one sold in 2010?

Mathematically, the One-Way ANOVA separates the total variance in the log-transformed SalePrice into two components: the variance explained by the categorical groups (Between-Group Variance) and the unexplained residual variance (Within-Group Variance). For example, if you group SalePrice by the year the house were sold, the ANOVA checks how much price difference is caused by the years (like 2006 vs. 2010) versus how much price difference is just random variation between two houses that are both sold in 2006.

Let $k$ be the total number of levels (groups) for a given feature, $n_i$ be the number of houses in group $i$, and $N$ be the total number of observations. For a house $j$ in group $i$ with price $y_{ij}$, we calculate the following terms:

* **Total Sum of Squares (SST):** Measures the total variation of all houses around the overall grand mean $\bar{y}_{..}$.
$$SST = \sum_{i=1}^{k} \sum_{j=1}^{n_i} (y_{ij} - \bar{y}_{..})^2$$
* **Sum of Squares Between (SSB):** Measures the variation of the group means $\bar{y}_{i.}$ around the grand mean, weighted by the group size. This represents the variance explained by the feature (Between-Group Variance).
$$SSB = \sum_{i=1}^{k} n_i (\bar{y}_{i.} - \bar{y}_{..})^2$$
* **Sum of Squares Within (SSW):** Measures the unexplained variation of individual houses around their respective group means (Within-Group Variance).
$$SSW = \sum_{i=1}^{k} \sum_{j=1}^{n_i} (y_{ij} - \bar{y}_{i.})^2$$

By definition, $SST = SSB + SSW$. To test whether the variation explained by the groups is statistically significant compared to the random noise, we compute the test statistic ($F$-ratio). This compares the Mean Square Between (MSB) to the Mean Square Within (MSW) by dividing the sums of squares by their respective degrees of freedom:
$$F = \frac{MSB}{MSW} = \frac{SSB / (k - 1)}{SSW / (N - k)}$$

In our code, the data was grouped by each feature's categorical levels, and the `scipy.stats.f_oneway` function was applied to automatically evaluate this F-statistic and output a corresponding p-value. A significance threshold of $\alpha = 0.05$ was used.

The results were clear: 7 of the tested features yielded good statistical significance ($p < 0.05$), rejecting our null hypothesis. The significant features and their exact p-values are:
* `OverallQual` ($p \approx 0.0$)
* `ExterQual` ($p = 6.93 \times 10^{-195}$)
* `BsmtQual` ($p = 1.48 \times 10^{-175}$)
* `KitchenQual` ($p = 4.44 \times 10^{-187}$)
* `FireplaceQu` ($p = 6.41 \times 10^{-20}$)
* `CentralAir` ($p = 9.86 \times 10^{-44}$)
* `LotShape` ($p = 7.86 \times 10^{-29}$)

Baseline quality and core physical characteristics drive the variance in housing price, while temporal features such as `MoSold` and `YrSold` show no significant impact. Consequently, these seven main effects were extracted to evaluate interactions.

### 4.3 Two-Way ANOVA and Interactions
One-Way ANOVA looks at house features one by one, but real estate value actually comes from how those features work together. To investigate this, we conducted a Two-Way ANOVA on all pairwise combinations of the 7 significant features identified in the previous step.

To ensure statistical rigor and avoid overfitting, a custom computational pipeline was developed with the following mechanisms:

**1. The Sparsity Filter:**
When cross-tabulating two categorical variables, many cells in the resulting matrix may contain few or no observations. Fitting a model to these sparse intersections causes it to overfit to outliers, creating artificial or "fake" statistical significance.

To mitigate this, a filter was applied before modeling. For every pair of features, a contingency table was created. If more than 75% of the interaction combinations contained fewer than 5 houses, the interaction was deemed too noisy and excluded from the analysis. This ensured our models were only trained on robust feature combinations.

**2. Type II ANOVA Modeling:**
For the combinations that passed the sparsity filter, an Ordinary Least Squares (OLS) regression model was fit using the `statsmodels` library, formulated as:
$$\text{LogSalePrice} \sim \text{Feature}_1 + \text{Feature}_2 + (\text{Feature}_1 \times \text{Feature}_2)$$
We then extracted the p-values from the generated **Type II ANOVA** table. Because of the number of houses existing per interaction is inconsistent, the order in which features are introduced to the model can severely bias the results. Utilizing a Type II sum of squares ensures that the main effects and the interaction term are evaluated independently of their input order, providing a more mathematically sound p-value for the interaction term.

**Results of the Interaction Analysis:**
Our analysis successfully identified 10 significant interactions, where the p-value was $< 0.05$, highlighting the complex nature of housing features. Certain quality features work together to boost (or lower) the final sale price. Notable combinations include:
* `ExterQual` $\times$ `CentralAir` ($p \approx 0.0$)
* `FireplaceQu` $\times$ `CentralAir` ($p = 2.71 \times 10^{-213}$)
* `ExterQual` $\times$ `BsmtQual` ($p = 9.97 \times 10^{-58}$)
* `BsmtQual` $\times$ `KitchenQual` ($p = 9.41 \times 10^{-35}$)

Other significant interactions involved `LotShape` combining with `OverallQual`, `ExterQual`, `BsmtQual`, and `KitchenQual`. For instance, the value added by a premium fireplace increases substantially when the home also features a high-quality central air system.

## 5. $2^k$ Factorial Design

### 5.1 Factor Selection and Binarization
For this part we picked 3 factors from the 7 that were significant in the ANOVA part: `OverallQual`, `KitchenQual` and `CentralAir`. We went with $k=3$ because that gives us $2^3 = 8$ combinations, which is rich enough to study interactions without things getting too complicated.

The issue is that factorial design needs binary factors, so we had to binarize two of them. For `OverallQual` we split at 5: above is *High*, below is *Low*. For `KitchenQual` we grouped `Ex` and `Gd` as *High* and everything else (`TA`, `Fa`, `Po`) as *Low*. For `CentralAir` there was nothing to do, it's already `Y` or `N`.

### 5.2 Main Effects and Interactions
We then computed the mean `SalePrice_log` for each of the 8 combinations, and fitted a full OLS model with the 3 main effects, 3 two-way interactions and the triple interaction. We use Type II like in the ANOVA part so that the order we put the variables in doesn't affect the results.

All 3 main effects are significant, which is consistent with what we found in ANOVA. The interesting part is the interactions: `OverallQual` $\times$ `KitchenQual` is significant ($p = 10^{-10}$), meaning the effect of kitchen quality on price is not the same depending on whether the house has high or low overall quality. In other words, the price gap between a good and a bad kitchen is more pronounced in a high overall quality home.

On the other hand the interactions with `CentralAir` are not significant ($p = 0.20$ with `OverallQual` and $p = 0.63$ with `KitchenQual`). So the AC acts independently — its effect on price is the same regardless of the quality of the house or the kitchen.

The interaction plots confirm this visually: the lines converge for `OverallQual` $\times$ `KitchenQual` (real interaction), and are parallel for the other two pairs (`OverallQual` $\times$ `CentralAir` and `KitchenQual` $\times$ `CentralAir`), which confirms the absence of interaction.

## 6. Parametric Regression
***Note:** Following the feedback received during the oral presentation, we updated the preprocessing pipeline to be consistent with Part 5.*

### 6.1 Model Specification and Preprocessing
To build our regression model, we took the 7 significant features from Part 2: `OverallQual`, `ExterQual`, `BsmtQual`, `KitchenQual`, `FireplaceQu`, `CentralAir` and `LotShape`. We added two continuous numerical features on top: `GrLivArea` (living area) and `TotalBsmtSF` (basement area). These two couldn't be tested in ANOVA because that method needs categorical groups, not continuous values.

To keep things consistent with Part 5, we applied the same preprocessing pipeline:
* Removed the same 2 extreme outliers (`GrLivArea` $> 4000\ ft^2$ and `SalePrice` $<$ $300,000) — 1458 observations remaining
* Applied `log1p` transform to `GrLivArea` and `TotalBsmtSF` to reduce skewness
* Encoded ordinal quality features as integers: `Po`=1, `Fa`=2, `TA`=3, `Gd`=4, `Ex`=5, `NA`=0 for missing basement or fireplace
* Used `SimpleImputer` with median strategy for remaining missing values

One thing worth noting: because `GrLivArea` is now log-transformed, its coefficient is a *log-log elasticity*, not a per-$ft^2$ effect. We'll explain what this means in section 6.4.

### 6.2 Regression Results

#### OLS
We fitted an OLS model with the 9 features and a constant for the intercept. The model gets an $R^2 = 0.832$ (adjusted $= 0.831$), meaning it explains 83.2% of the variance in price. The global F-statistic is $796.8$ with $p = 0.00$, so the model is globally significant.

Looking at the coefficients converted to the original scale with $e^{\beta} - 1$:
* `GrLivArea` : $+54\%$ (log-log elasticity)
* `CentralAir` : $+21\%$ — biggest per-unit effect among binary features
* `OverallQual` : $+9\%$ per level
* `KitchenQual` : $+8\%$ per level
* `ExterQual` : $+6\%$ per level
* `BsmtQual` : $+5\%$ per level
* `LotShape` : $-5\%$ — only feature with a negative effect
* `FireplaceQu` : $+2\%$ per level
* `TotalBsmtSF` : not significant ($p = 0.259$, $F = 1.28$)

We noticed that `TotalBsmtSF` is no longer significant, which was surprising at first. After thinking about it, we think it's because of multicollinearity: after log-transforming both `GrLivArea` and `TotalBsmtSF`, they became correlated — big house = big basement. Once `GrLivArea` is already in the model, `TotalBsmtSF` doesn't add any new information so its coefficient becomes noisy and statistically meaningless.

#### Model Diagnostics
To check that the model is valid we plotted three plots: residuals vs fitted values, a Q-Q plot of the residuals, and a residuals histogram.

The residuals vs fitted plot shows the points are mostly centered around 0, so the model isn't biased. The spread looks fairly homogeneous across the range of fitted values, which is a good sign for homoscedasticity.

The Q-Q plot shows the residuals follow the normal line well in the middle, but there is still some deviation at the lower tail. This left-side deviation corresponds to a cluster of houses the model consistently overestimates — probably houses with unusual characteristics that our 9 features don't fully capture.

The residuals histogram confirms this: the distribution looks like a bell curve centered at 0 but with a noticeable left tail. After removing the two extreme outliers, the model improved from $R^2 = 0.816$ to $R^2 = 0.832$, so the decision was clearly the right one.

#### Regularization
OLS gives a good fit but doesn't penalize model complexity. We wanted to check whether Ridge or Lasso could improve generalization and help mitigate multicollinearity, so we tested both with different alpha values on a 20% validation set.

| Model | $R^2$ | RMSE |
| :--- | :--- | :--- |
| OLS | 0.8284 | 0.1701 |
| Ridge $\alpha=0.01$ | 0.8284 | 0.1701 |
| Ridge $\alpha=10$ | 0.8285 | 0.1700 |
| Lasso $\alpha=0.01$ | 0.8275 | 0.1705 |
| Lasso $\alpha=0.1$ | 0.7165 | --- |
| Lasso $\alpha=1.0$ | $\approx 0.0$ | --- |

The validation $R^2$ (0.8284) is slightly lower than the training $R^2$ (0.832), which is the expected behavior — the model performs a bit better on data it has seen. The gap is very small though, so there is almost no overfitting. With only 9 features the model stays simple and generalizes well.

Ridge has basically no effect regardless of alpha — the $R^2$ stays at $0.8284$–$0.8285$ for every value we tested, identical to OLS. With only 9 features the model is already well-conditioned so the $\beta^2$ penalty doesn't change anything.

Lasso is a different story. It works fine at $\alpha = 0.01$ but degrades quickly after that. At $\alpha = 0.1$ the $R^2$ drops to $0.7165$, and at $\alpha = 1.0$ the model completely collapses. The reason is that Lasso penalizes $|\beta|$ instead of $\beta^2$ — for a small coefficient like $0.05$, Ridge penalizes $0.05^2 = 0.0025$ but Lasso penalizes $|0.05| = 0.05$, which is 20x more aggressive. So it kills the coefficients way faster for the same alpha.

Overall OLS is still the best here. With only 9 features there is nothing to regularize, so Ridge and Lasso don't add anything.

### 6.3 ANOVA on the OLS model
To measure the individual contribution of each feature, we applied a Type II ANOVA on the fitted OLS model. The top 3 are `GrLivArea` ($F = 624$), `OverallQual` ($F = 212$) and `CentralAir` ($F = 102$). `TotalBsmtSF` has $F = 1.28$ with $p = 0.259$, confirming it contributes nothing once `GrLivArea` is already in the model. All other features are significant ($p < 0.05$).

### 6.4 Coefficient Interpretation
After the log-transform, `GrLivArea` now has the biggest coefficient (0.4322) and the highest F-statistic ($F = 624$), so at this point it's clear that it's the most important feature.

Our first instinct was to just rank the features by their coefficient, but then we realized that doesn't really make sense because they don't all vary the same amount in the data. Take `CentralAir` for example — its coefficient is 0.1882 which looks big, but 93.5% of houses already have AC. So almost every house in the dataset has it, which means the feature barely changes between houses. A feature that doesn't vary can't really explain much even if its coefficient looks large.

So we computed the realistic total effect over the actual range of each feature using $e^{\beta \times \text{range}} - 1$:
* `GrLivArea` : $\log(\text{area})$ goes from 7 to 8 for 95% of houses (range = 1 log-unit) $\rightarrow$ $e^{0.4322 \times 1} - 1 = \mathbf{+54\%}$
* `OverallQual` : quality goes from 4 to 9 for 95% of houses (range = 5 levels) $\rightarrow$ $e^{0.0847 \times 5} - 1 = \mathbf{+53\%}$
* `CentralAir` : 93.5% already have AC (almost no variation) $\rightarrow$ $e^{0.1882 \times 1} - 1 = \mathbf{+21\%}$

Honestly we didn't expect `GrLivArea` and `OverallQual` to be that close in total effect ($+54\%$ vs $+53\%$). But `GrLivArea` still has a way higher F-statistic (624 vs 212). We think it's because living area is something concrete and objective — every $ft^2$ adds value in a predictable way so the model makes fewer errors. `OverallQual` is more subjective, two houses rated 7 can sell at very different prices depending on things the score doesn't capture, so there is more noise.

We can't just look at the coefficient alone. We also have to check how much the feature actually varies in the data and how consistently the model predicts from it. After the log-transform, `GrLivArea` is clearly the strongest feature on all of these criteria.

## 7. Neural-Network Regression

### 7.1 Model Inputs and Preprocessing
To squeeze out the best possible performance for our Multi-Layer Perceptron (MLP), we didn't just throw the raw data at the model, because that would make it difficult for the model to learn properly. So we implemented a transformation pipeline. First, we dropped extreme outliers (houses with over 4,000 sq ft but priced under $300,000) to prevent our network from learning these really bad weights.

From there, we "engineered" several high-value features:
* **Feature Aggregation:** We calculated `TotalSF` (combining basement, 1st, and 2nd floors) and `TotalBath`.
* **Temporal Features:** We calculated the true age of the house at sale (`HouseAge`) and the years since remodeling (`RemodAge`).
* **Super-Variables:** We created interaction terms like `TotalQual` (OverallQual $\times$ OverallCond) and `Qual_x_SF` to capture exponential value increases in large, high-quality homes.
* **Manual-Mapping:** Instead of blindly one-hot encoding everything, we manually mapped ordinal variables (like `KitchenQual` and `BsmtQual`) to integer scales (1 to 5), preserving their natural hierarchy.

These transformations were done in order to alleviate the mathematical and computational strains on the model.

Finally, we split the feature space to apply specific, manual transformations to the numerical and categorical data before feeding them into the network:
* **Numerical Processing:** Missing values were replaced using the median of each respective feature. We then applied standard scaling ($z = \frac{x - \mu}{\sigma}$) to force the numerical features to behave and have a mean of 0 and a standard deviation of 1. This ensures all variables are on the same scale, allowing the model's weights to update smoothly.
* **Categorical Processing:** Missing categorical values were filled with a constant string ('Missing') so the model could recognize absence as its own distinct category. We then applied One-Hot Encoding, generating a binary matrix all while ignoring unknown test categories to prevent errors.
* **Matrix Assembly:** The standardized numerical array and the encoded categorical array were horizontally concatenated (using NumPy's `hstack`) to construct the final matrix passed to the MLP Regressor.

Finally, we handled the skewness of our surface area features using a log transformation ($\log(1+x)$). Missing numerical values were imputed with the median and standardized using `StandardScaler`, while remaining categoricals were imputed with a constant ('Missing') and processed via `OneHotEncoder`. Everything was fused into a single dense matrix using `NumPy`'s horizontal stack.

### 7.2 Training and Validation
Because neural networks are highly sensitive to their architecture, we initially set up a massive `GridSearchCV` with 5-fold cross-validation to explore different layer sizes and regularization penalties.

Through this search, we learned exactly what our data needed! We discovered that a two-layer architecture with **60 neurons in the first layer and 30 in the second** provided the perfect balance—giving the model enough capacity to learn complex patterns without memorizing the noise. We also found that our feature engineering required a heavy L2 regularization penalty (**alpha = 5.0**) to prevent overfitting.

We originally attempted to build and train this network in PyTorch using the Adam optimizer, but our initial results weren't good enough. Adam heavily struggled to converge properly on this specific dataset. Pivoting to `scikit-learn` and utilizing the `lbfgs` quasi-Newton optimizer completely turned things around, as it is structurally far better suited for converging on smaller datasets by using the actual curvature of the loss function allowing it to take precise, calculated steps toward the minimum. L-BFGS provides a noticeably smoother and more efficient convergence trajectory for this type of problem compared to standard first-order algorithms like Adam. We paired this solver with the `relu` activation function for optimal performance.

Once we learned these optimal parameters, we didn't want to force our final notebook to re-run an expensive grid search every single time. Instead, we took those winning parameters and hardcoded them into a streamlined, final `MLPRegressor` model. We gave it a generous `max_iter=500` to ensure it fully converged, and then fit it directly on our fully processed training matrix. Checking our training RMSE on the log scale confirmed that the model learned the underlying patterns beautifully, being under 0.07!

### 7.3 Evaluation and Residual Analysis
During our first evaluation pipeline, we hit a brief technical issue (a `ValueError` regarding string conversion). We quickly realized we were mistakenly passing the raw, unencoded `X_train` dataframe to the model. Once we fed our fully scaled and one-hot encoded `X_train_processed` matrix into the optimized MLP, the results were quite good.

Evaluating the training set, the model achieved a massive $R^2$ score, meaning it successfully captured the vast majority of the variance in the log-transformed sale prices. Translating the RMSE back to the original dollar scale showed that our prediction error represents a very reasonable percentage of the mean actual house price.

To visualize our model "credibility", we generated a 2x2 grid of evaluation plots:

* **Predicted vs. Actual (Log & Original Scales):** The scatter points follow the perfect prediction line (indicated by the dashed lines). The model demonstrates notable predictive power without automatically overvaluing or undervaluing homes across different price levels.
* **Residuals vs. Fitted:** The residuals are randomly scattered around the horizontal zero line. Crucially, there are no obvious patterns or shapes, thus indicating homoscedasticity. Our model captures the underlying variance without leaving obvious non-linear trends behind.
* **Residual Distribution:** The histogram reveals a beautiful, bell-shaped normal distribution perfectly centered at zero. This provides ultimate confirmation that our initial choice to log-transform the target variable in Part 1 was absolutely necessary and effective.

### 7.4 Kaggle Submission File
Since our target variable was log-transformed at the start of the pipeline to ensure normal distribution of errors, the final predictions output by the MLP were also in log space.

To generate our Kaggle submission, we applied the inverse transformation using `NumPy`'s `expm1` function to bring our predictions back to the original dollar scale. We mapped these predicted prices back to their respective test set `Id`s and exported them to `submission.csv`. Given our strong CV score and clean residuals, this model positions us perfectly to hit our goal of breaking the 0.13 RMSE barrier on the public leaderboard!

## References

1. CStat26 Project Instructions. [Link](https://github.com/lydiaYchen/CStat26/tree/main/Project)
2. Kaggle. House Prices: Advanced Regression Techniques. [Link](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
3. ResearchGate. A comparison of the performance of the Adam optimizer. [Link](https://www.researchgate.net/figure/A-comparison-of-the-performance-of-the-Adam-optimizer-an-algorithm-for-first-order_fig3_360640362)