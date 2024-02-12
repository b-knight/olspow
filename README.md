# olspow: Power Analysis for Experiments Using Regression / Clustered Data
## What is it?
**olspow** is a python package designed to elucidate the level of statistical power, the sample size, and the minimum detectable effect (MDE) within
the context of randomized, controlled trials (i.e. A/B tests) where the experimenter is using OLS to estimate the effect size of a dichotomous
treatment variable. The underlying methodology can be equally applied to clustered data, or simpler experimental designs where the relationship
between observations and units of experimental assignment are 1:1.

## Why OLS?
In the context of an A/B test, the mean difference across different covariates will be zero assuming that experimental assignment is appropriately
random. However, the observed difference will rarely be precisely zero. These non-zero differences (typically referred to as 'covariate imbalance')
introduce noise within our estimate of the effect of being treated on the response variable.<br>

Ordinary Least Squares (OLS) is a well understood estimator
that is available in a variety of packages in Python (*[scipy](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.linregress.html)*,
*[statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html)*,
*[sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)*) which is well-suited to mitigate covariate
imbalance assuming that the experimenter provides appropriate covariates to adjust for (i.e. predictors that are orthogonal to the treatment variable).
In other words, thoughtful use of OLS is a form of covariate adjustment. Most power analysis tools presume the use of a
*[t-test](https://www.statsmodels.org/stable/generated/statsmodels.stats.power.tt_ind_solve_power.html)*
and as such, cannot account for the degree to which covariate adjustment improves the sensitivity of our statistical test. **olspow** was specifically
designed to address this problem.


## Calling the solve_power() Method
All functionality is accessed via the **solve_power()** method, which returns the minimum detectable effect (MDE), power, or required sample size
(as measured in number of units of experimental assignment).<br>

**solve_power**(data, endog, exog, cluster, ratio, alpha, mde, power, n, alternative, verbose):
> **_is_ratio (boolean, required)_** : A boolean representing whether the metric represents a ratio of two variables. Setting this value to 'True' makes the 'numerator' and 'denominator' arguments mandatory while rendering the 'endog' argument
superfluous. Contrariwise, a value of 'False' will obviate the need for the 'numerator' and 'denominator' arguments while making the 'endog' argument mandatory.<br>
> **_data (pandas.core.frame.DataFrame, required)_** : A Pandas dataframe containing (at a minimum) historical values for the response variable and the cluster key.
>  Ideally, the dataframe contains 100 unique units of experimental assignment per covariate.<br>
> **_endog (string, optional)_** : A string representing the response variable being modeled - i.e. the metric of interest.<br>
> **_exog (list of strings, required)_** : A list of strings representing the names of covariates that are being adjusted for (i.e. column names within the Pandas dataframe).<br>
> **_numerator (str, optional)_**: The name of the numerator variable (e.g. if the metric is items fulfilled per unit of time, then this value would be the number of items fulfilled). Required when 'is_ratio' = True. Defaults to None.<br>
> **_denominator (str, optional)_**: The name of the denominator variable (e.g. if the metric is items fulfilled per unit of time, then this value would be the amount of time). Required when 'is_ratio' = True. Defaults to None.<br>
> **_cluster (string, required)_** : The name of the column in the Pandas dataframe that serves as the cluster key (unit of experimental assignment)<br>
> **_ratio (float, optional)_** : Assumed ratio of treated units of assignment to control units of assignment. Defaults to 0.5 (i.e. 50:50 assignment between treated and control)<br>
> **_alpha (float, optional)_** : A float corresponding to the false positive rate. The default value is 0.05<br>
> **_power (float, optional)_** : A float corresponding to the level of statistical power.<br>
> **_n (integer, optional)_** : An integer corresponding to the number of units of experimental assignment.<br>
> **_alternative (string, optional)_** : A string that can be 'one-sided' or 'two-sided' denoting if the experimenter is designing a one-tailed or two-tailed test. The default value is 'two-sided'.<br>
> **_verbose (string, boolean)_** : A boolean that, if set to 'true', will print all steps in the power analysis workflow.<br>
