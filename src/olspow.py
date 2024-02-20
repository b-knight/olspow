import humanize
import math
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm

ABS_MIN_NOBS = 1000  # Absolute minimum number of total observations
NOBS_PER_COVARIATE = 200  # Recommended number of incremental nobs per covariate


def solve_power(
    is_ratio,
    data,
    exog,
    cluster,
    endog=None,
    numerator=None,
    denominator=None,
    n=None,
    mde=None,
    power=None,
    ratio=0.5,
    alpha=0.05,
    alternative="two-sided",
    verbose=False,
):
    """
    Solves for the statistical power of a hypothesis test in a power analysis.

    Args:
        is_ratio (bool): Indicates whether the response variable is a ratio metric.
        data (pandas.DataFrame): The dataset containing the response variable and exogenous variables.
        exog (list): The list of exogenous variables.
        cluster (str): The name of the cluster variable.
        endog (str, optional): The name of the endogenous variable. Defaults to None.
        numerator (str, optional): The name of the numerator variable. Defaults to None.
        denominator (str, optional): The name of the denominator variable. Defaults to None.
        n (int, optional): The number of observations to be used in the analysis. Defaults to None.
        mde (float, optional): The minimum detectable effect size. Defaults to None.
        power (float, optional): The desired level of statistical power. Defaults to None.
        ratio (float, optional): The proportion of assignment entities to be treated. Defaults to 0.5.
        alpha (float, optional): The significance level of the test. Defaults to 0.05.
        alternative (str, optional): The type of hypothesis test to be performed. Defaults to "two-sided".
        verbose (bool, optional): Indicates whether to print additional information. Defaults to False.

    Raises:
        ValueError: If any of the input values are invalid.

    Returns:
        None
    """

    def _validate_number_of_inputs():
        """
        Validates the number of inputs provided for the function.

        Raises:
            ValueError: If the number of supplied arguments is not equal to 2.
        """
        supplied_args = 0
        for arg in [n, mde, power]:
            if arg is not None:
                supplied_args += 1
        if supplied_args != 2:
            raise ValueError(
                "Two (and only two) of the three arguments : 'n', "
                + "'mde', and 'power' may be provided."
            )
        if endog is None and (numerator is None or denominator is None):
            raise ValueError(
                "Must provide either the 'endog' argument or both the "
                + "'numerator' and 'denominator' arguments."
            )
        if endog is not None and (numerator is not None and denominator is not None):
            raise ValueError(
                "Cannot provide both the 'endog' argument and the "
                + "'numerator' and 'denominator' arguments. If the response "
                + "variable is a ratio metric, set the 'is_ratio' argument "
                + "to 'True' and provide the 'numerator' and 'denominator' "
                + "arguments while omitting the 'endog' argument. If the "
                + "response variable is not a ratio metric, set the 'is_ratio' "
                + "argument to 'False' and provide the 'endog' argument while "
                + "omitting the 'numerator' and 'denominator' arguments."
            )

    def _validate_input_values():
        """
        Validates the input values for the OLSPOW analysis.

        Raises:
            ValueError: If any of the input values are invalid.
        """
        if is_ratio not in [True, False]:
            raise ValueError(
                "The 'is_ratio' argument must be a boolean value "
                + "indicating whether the response variable is a "
                + "ratio of two variables."
            )
        if is_ratio and (numerator is None or denominator is None):
            raise ValueError(
                "The 'numerator' and 'denominator' arguments must both be "
                + "provided when the response variable is a ratio of two variables. "
                + "Please provide the appropriate column names within the dataframe."
            )
        if not isinstance(data, pd.DataFrame):
            raise ValueError("The 'data' argument must be a pandas DataFrame.")
        if endog is not None and not isinstance(endog, str):
            raise ValueError(
                "The 'endog' argument must be a string corresponding to the "
                + "name of the response variable."
            )
        if endog is not None and endog.lower() not in [x.lower() for x in data.columns]:
            raise ValueError(
                f"The response variable '{endog}' was not found within "
                + "the provided Pandas dataframe."
            )
        if not isinstance(exog, list):
            raise ValueError(
                "The 'exog' argument must be a list of strings corresponding "
                + "to the names of the predictor variables."
            )
        if len(exog) == 0:
            if verbose:
                print(
                    "No covariates were specified. Power analysis will be "
                    + "performed using only the intercept term."
                )
                print(
                    "NOTE: With no covariates, estimation will be akin to a "
                    + "t-test using pooled variance (i.e. a two-sample t-test). "
                    + "\nAre you sure this is the desired model specification?"
                )
        for predictor in exog:
            if not isinstance(predictor, str):
                raise ValueError(
                    "The 'exog' argument must be a list of strings "
                    + "corresponding to the names of the predictor variables."
                )
            if predictor.lower() not in [x.lower() for x in data.columns]:
                raise ValueError(
                    f"The predictor variable '{predictor}' was not found "
                    + "within the provided Pandas dataframe."
                )
            if is_ratio and (
                predictor.lower() == numerator.lower()
                or predictor.lower() == denominator.lower()
            ):
                raise ValueError(
                    "The 'exog' argument must not contain the same variables as the "
                    + "numerator or denominator of the target ratio metric."
                )
        if not isinstance(cluster, str):
            raise ValueError(
                "The 'cluster' argument must be a string corresponding to the "
                + "name of the entity to receive experimental assignment."
            )
        if numerator is not None and numerator.lower() not in [
            x.lower() for x in data.columns
        ]:
            raise ValueError(
                f"The numerator of the target ratio metric -'{numerator}' - "
                + "was not found within the provided Pandas dataframe."
            )
        if numerator is not None and denominator is None and not is_ratio:
            print(
                "WARNING: The 'numerator' argument was provided, but "
                + "the response variable was defined as not being a "
                + "ratio of two variables. The 'numerator' argument will "
                + "be ignored."
            )
        if denominator is not None and denominator.lower() not in [
            x.lower() for x in data.columns
        ]:
            raise ValueError(
                f"The denominator of the target ratio metric -'{numerator}' - "
                + "was not found within the provided Pandas dataframe."
            )
        if denominator is not None and numerator is None and not is_ratio:
            print(
                "WARNING: The 'denominator' argument was provided, but "
                + "the response variable was defined as not being a "
                + "ratio of two variables. The 'denominator' argument will "
                + "be ignored."
            )
        if denominator is not None and numerator is not None and not is_ratio:
            raise ValueError(
                "The 'numerator' and 'denominator' arguments were provided, "
                + "but the response variable was defined as not being a ratio "
                + "of two variables. Please set the 'is_ratio' argument to 'True'."
            )
        if n is not None and not isinstance(n, int):
            raise ValueError(
                "The 'n' argument must be an integer corresponding to the "
                + "number of observations to be used in the analysis."
            )
        if not (isinstance(alpha, float) or (alpha < 0.0 or alpha > 1.0)):
            raise ValueError(
                "The 'alpha' argument must be a float between 0.0 and 1.0 "
                + "corresponding to the significance level of the test."
            )
        if (
            (power is not None and not isinstance(power, float))
            or (isinstance(power, float) and power < 0.0)
            or (isinstance(power, float) and power > 1.0)
        ):
            raise ValueError(
                "The 'power' argument must be a float between 0.0 and 1.0 "
                + "corresponding to the desired level of statistical power."
            )
        if not isinstance(ratio, float) or ratio < 0.0 or ratio > 1.0:
            raise ValueError(
                "The 'ratio' argument must be a float between 0.0 and 1.0 "
                + "corresponding to the proportion of assignment entities"
                + "to be treated. A value of 0.5 is typically optimal."
            )
        if alternative not in ["two-sided", "one-sided"]:
            raise ValueError(
                "The 'alternative' argument must be a string "
                + "corresponding to the type of hypothesis test to "
                + "be performed."
            )

    def _aggregate_data(
        data, endog, exog, cluster, numerator, denominator, agg_method=None
    ):
        """
        Create aggregated data shape based on specified parameters.

        Args:
            data (pandas.DataFrame): The input data.
            endog (str): The name of the endogenous variable.
            exog (list): The list of exogenous variables.
            cluster (str): The name of the cluster variable.
            numerator (str): The name of the numerator variable.
            denominator (str): The name of the denominator variable.
            agg_method (dict, optional): The aggregation method for each variable. Defaults to None.

        Returns:
            pandas.DataFrame: The aggregated data shape.

        """
        if verbose:
            print(f"Aggregating data on the cluster key: {cluster}")
        if agg_method is None:
            agg_method = {}
            if numerator is None and denominator is None:
                agg_method[endog], agg_method[cluster] = (
                    "sum",
                    "count",
                )
            else:
                agg_method[numerator], agg_method[denominator], agg_method[cluster] = (
                    "sum",
                    "sum",
                    "count",
                )
            for predictor in exog:
                agg_method[predictor] = "sum"
        data_agg_out = data.groupby(cluster).agg(agg_method)
        data_agg_out.rename(columns={cluster: "NOBS_IN_CLUSTER"}, inplace=True)
        if verbose:
            print(f"{len(data_agg_out)} clusters were found in the data.")
        return data_agg_out

    def _validate_sample_size(data, exog):
        """
        Validates the sample size post aggregation.

        Args:
            data (pandas.DataFrame): The post-aggregation data.

        Returns:
            None
        """
        num_coef = len(exog) + 1
        recommended_nobs = ABS_MIN_NOBS + (num_coef * NOBS_PER_COVARIATE)
        if len(data) < recommended_nobs:
            print(
                f"NOTE: Model specification implies {num_coef} covariates "
                + f"({num_coef} coefficient estimate(s)). \nHowever, only "
                + f"{humanize.intcomma(len(data))} unique units of assignment "
                + "were found. \nConsider increasing the size of the "
                + "historical dataset to at least "
                + f"{humanize.intcomma(recommended_nobs)} "
                + "units of assignment per coefficient estimate."
            )
        if len(data) < num_coef:
            raise ValueError(
                f"The number of units of assignment in the historical dataset "
                + f"({len(data)}) is insufficient to estimate the model. "
                + f"\nTry increasing the size of the historical dataset to at "
                + f"least {humanize.intcomma(recommended_nobs)} units "
                + "of assignment."
            )

    def _compute_residuals_of_response_var(is_ratio, data, endog, exog):
        """
        Compute the residuals of the response variable.

        Parameters:
        - is_ratio (bool): Indicates whether the response variable is a ratio metric.
        - data (pandas.DataFrame): The dataset containing the response variable and exogenous variables.
        - endog (str): The name of the response variable column in the dataset.
        - exog (str): The name of the exogenous variable column in the dataset.

        Returns:
        - residuals (numpy.ndarray): The residuals of the response variable after regression.
        """

        if verbose:
            print("Computing the residuals of the response variable.")
        if is_ratio:
            y = data[numerator]
        else:
            y = data[endog]
        X = data[exog]
        X = sm.add_constant(X)
        model = sm.OLS(y, X)
        model_fit = model.fit()
        if verbose:
            print("\nThe response variable was regressed on the exogenous variables.")
            print("-" * 78)
            print(model_fit.summary())
        return model_fit.resid

    def _compute_residuals_of_denominator(is_ratio, data, exog):
        """
        Compute the residuals corresponding to the number of observations per cluster.

        Parameters:
        - is_ratio (bool): Indicates whether the dependent variable is a ratio metric.
        - data (pandas.DataFrame): The data containing the variables.
        - exog (str or list): The exogenous variable(s) used in the regression.

        Returns:
        - pandas.Series: The residuals of the regression model.
        """
        if verbose:
            print(
                "\nComputing the residuals corresponding to the "
                + "number of observation per cluster."
            )
        X = data[exog]
        if is_ratio:
            y = data[denominator]
        else:
            y = data["NOBS_IN_CLUSTER"]
        X = sm.add_constant(X)
        model = sm.OLS(y, X)
        model_fit = model.fit()
        if verbose:
            print(
                "The number of observations per cluster was regressed on "
                + "the exogenous variables."
            )
            print("-" * 78)
            print(model_fit.summary())
        return model_fit.resid

    def _compute_ratio_of_means(is_ratio, working_df, endog):
        """
        Compute the ratio of means based on the given parameters.

        Parameters:
        - is_ratio (bool): Flag indicating whether the response variable is a ratio metric.
        - working_df (pandas.DataFrame): The DataFrame containing the data.
        - endog (str): The name of the endogenous variable.

        Returns:
        - float: The computed ratio of means.
        """
        if is_ratio:
            return working_df[numerator].mean() / working_df[denominator].mean()
        else:
            return (
                working_df[endog].mean() / working_df["NOBS_IN_CLUSTER"].mean(),
                working_df[endog].mean(),
                working_df["NOBS_IN_CLUSTER"].mean(),
            )

    def _compute_sample_stdev(is_ratio, working_df, endog, exog):
        """
        Compute the standard deviation of the double residuals.

        Parameters:
        - is_ratio (bool): Indicates whether the response variable is a ratio.
        - working_df (pandas.DataFrame): The working dataset.
        - endog (str): The name of the response variable.
        - exog (list): The names of the explanatory variables.

        Returns:
        - float: The standard deviation of the double residuals.
        """
        n = len(working_df)
        dof = n - (len(exog) + 1)
        y_resid = _compute_residuals_of_response_var(is_ratio, working_df, endog, exog)
        w_resid = _compute_residuals_of_denominator(is_ratio, working_df, exog)
        theta, y_bar, w_bar = _compute_ratio_of_means(is_ratio, working_df, endog)
        sample_std = np.sqrt(((y_resid - theta * w_resid) ** 2).sum() / dof)
        return (
            sample_std,
            dof,
            theta,
            y_bar,
            w_bar,
            y_resid,
            w_resid,
        )

    def _fetch_z_statistic(input, alternative="one-sided"):
        """
        Fetches the z-statistic based on the input and alternative.

        Parameters:
        input (float): The input value.
        alternative (str): The alternative hypothesis ("one-sided" or "two-sided").

        Returns:
        tuple: A tuple containing the z-statistic and the modified input value.
        """
        if alternative == "one-sided":
            return abs(st.norm.ppf(input)), input
        if alternative == "two-sided":
            return abs(st.norm.ppf(input / 2)), input / 2

    def _derive_mde(
        is_ratio_metric,
        sample_size,
        historical_stdev,
        w_bar,
        assignment_ratio,
        alpha_value,
        power_value,
        num_sides,
        covariates,
    ):
        """
        Derives the minimum detectable effect (MDE) for power analysis.

        Args:
            is_ratio_metric (bool): Indicates whether the response variable is a ratio metric.
            sample_size (int): The sample size.
            historical_stdev (float): The historical standard deviation.
            w_bar (float): The average treatment effect.
            assignment_ratio (float): The ratio of treatment assignment.
            alpha_value (float): The significance level.
            power_value (float): The desired power.
            num_sides (int): The number of sides for the hypothesis test.
            covariates (list): The list of covariates.

        Returns:
            float: The minimum detectable effect (MDE) for power analysis.
        """

        if verbose:
            print("Deriving the minimum detectable effect.")
        alpha_z, alpha = _fetch_z_statistic(alpha_value, num_sides)
        beta_z, beta = _fetch_z_statistic(1.0 - power_value)
        mde = (
            (alpha_z + beta_z)
            * np.sqrt((1 / assignment_ratio) + (1 / (1 - assignment_ratio)))
            * (historical_stdev / (w_bar * np.sqrt(sample_size)))
        )
        if verbose:
            print("-" * 80)
        if is_ratio_metric:
            variable_type = "ratio metric"
        else:
            variable_type = "non-ratio metric"
        if verbose:
            print(
                f"Power analysis where the response variable is a {variable_type} is as follows:"
            )
            print(
                f"The minimum detectable effect (MDE) was estimated as {humanize.intcomma(mde)}. "
                + f"\nEstimate was based on {humanize.intcomma(len(data))} observations "
                + f"across {humanize.intcomma(len(working_df))} clusters of historical data. "
                + f"\nThe model specification used {humanize.intcomma(len(covariates))} "
                + f"covariates, assumes {humanize.intcomma(len(covariates)+1)} coefficient "
                + f"estimates (alpha={round(alpha, 4)}, beta={round(beta, 4)})."
            )
        return mde, alpha_z, beta_z

    def _derive_n(
        is_ratio_metric,
        assumed_mde,
        historical_stdev,
        w_bar,
        assignment_ratio,
        alpha_value,
        power_value,
        num_sides,
        covariates,
    ):
        """
        Derives the required sample size for power analysis.

        Args:
            is_ratio_metric (bool): Indicates whether the response variable is a ratio metric.
            assumed_mde (float): Minimum detectable effect (MDE) for the power analysis.
            historical_stdev (float): Standard deviation of the historical data.
            w_bar (float): Average cluster size.
            assignment_ratio (float): Ratio of treatment assignment.
            alpha_value (float): Significance level (alpha) for the power analysis.
            power_value (float): Desired power value for the power analysis.
            num_sides (int): Number of sides for the hypothesis test.
            covariates (list): List of covariates used in the model specification.

        Returns:
            int: The minimum required sample size for the power analysis.
        """

        if verbose:
            print("Deriving the required sample size.")
        alpha_z, alpha = _fetch_z_statistic(alpha_value, num_sides)
        beta_z, beta = _fetch_z_statistic(1.0 - power_value)
        sample_size = math.ceil(
            (
                ((alpha_z + beta_z) ** 2)
                * ((1 / assignment_ratio) + 1 / (1 - assignment_ratio))
                * ((historical_stdev) ** 2 / ((w_bar**2) * (assumed_mde**2)))
            )
        )
        if verbose:
            print("-" * 80)
        if is_ratio_metric:
            variable_type = "ratio metric"
        else:
            variable_type = "non-ratio metric"
        if verbose:
            print(
                f"Power analysis where the response variable is a {variable_type} is as follows:"
            )
            print(
                "The minimum required sample size was estimated as "
                + f"{humanize.intcomma(sample_size)} using a minimum "
                + f"detectable effect (MDE) of {humanize.intcomma(assumed_mde)}. "
                + f"\nEstimate was based on {humanize.intcomma(len(data))} observations "
                + f"across {humanize.intcomma(len(working_df))} clusters of historical data. "
                + f"\nThe model specification used {humanize.intcomma(len(covariates))} "
                + f"covariates, assumes {humanize.intcomma(len(covariates)+1)} coefficient "
                + f"estimate(s) (alpha={round(alpha, 4)}, beta={round(beta, 4)})."
            )
        return sample_size, alpha_z, beta_z

    def _derive_power(
        is_ratio_metric,
        sample_size,
        assumed_mde,
        historical_stdev,
        alpha_value,
        assignment_ratio,
        w_bar,
        num_sides,
        covariates,
    ):
        """
        Calculates the estimated statistical power for a power analysis.

        Args:
            is_ratio_metric (bool): Indicates whether the response variable is a ratio metric.
            sample_size (int): The sample size.
            assumed_mde (float): The minimum detectable effect (MDE).
            historical_stdev (float): The historical standard deviation.
            alpha_value (float): The alpha value.
            assignment_ratio (float): The assignment ratio.
            w_bar (float): The value of w_bar.
            num_sides (int): The number of sides.
            covariates (list): The list of covariates.

        Returns:
            float: The estimated statistical power.
        """

        if verbose:
            print("Deriving the estimated statistical power.")
        alpha_z, alpha = _fetch_z_statistic(alpha_value, num_sides)
        est_power = st.norm.cdf(
            (
                assumed_mde
                / (
                    np.sqrt((1 / assignment_ratio) + 1 / (1 - assignment_ratio))
                    * (historical_stdev / (w_bar * np.sqrt(sample_size)))
                )
            )
            - alpha_z
        )
        if verbose:
            print("-" * 80)
        if is_ratio_metric:
            variable_type = "ratio metric"
        else:
            variable_type = "non-ratio metric"
        if verbose:
            print(
                f"Power analysis where the response variable is a {variable_type} is as follows:"
            )
            print(
                "Statistical power was estimated as "
                + f"{round(est_power, 4)} using a minimum "
                + f"detectable effect (MDE) of {humanize.intcomma(assumed_mde)}. "
                + f"\nEstimate was based on {humanize.intcomma(len(data))} observations "
                + f"across {humanize.intcomma(len(working_df))} clusters of historical data. "
                + f"\nThe model specification used {humanize.intcomma(len(covariates))} "
                + f"covariates, assumes {humanize.intcomma(len(covariates)+1)} coefficient "
                + f"estimates (alpha={round(alpha, 4)})."
            )
        return est_power, alpha_z

    def _create_diagnostics_dict(
        sample_stdev,
        dof,
        assignment_ratio,
        theta,
        y_bar,
        w_bar,
        y_residuals,
        w_residuals,
        hypothesis,
        nobs_pre_aggregation,
        nobs_post_aggregation,
        alpha_z=None,
        beta_z=None,
        mde_est=None,
        n_est=None,
        power_est=None,
    ):
        """
        Create a diagnostics dictionary with various statistics and values.

        Args:
            sample_stdev (float): The sample standard deviation.
            dof (int): The degrees of freedom.
            assignment_ratio (float): The assignment ratio.
            theta (float): The value of theta.
            y_bar (float): The numerator y-bar.
            w_bar (float): The denominator w-bar.
            y_residuals (list): The y-bar residuals.
            w_residuals (list): The w-bar residuals.
            hypothesis (str): The tails used in the t-test.
            nobs_pre_aggregation (int): The number of observations before aggregation.
            nobs_post_aggregation (int): The number of observations after aggregation.
            alpha_z (float, optional): The alpha z-statistic. Defaults to None.
            beta_z (float, optional): The beta z-statistic. Defaults to None.
            mde_est (float, optional): The estimated minimum detectable effect. Defaults to None.
            n_est (float, optional): The estimated required sample size. Defaults to None.
            power_est (float, optional): The estimated power. Defaults to None.

        Returns:
            dict: The diagnostics dictionary containing various statistics and values.
        """
        if verbose:
            print("Creating the diagnostics dictionary.")
        diagnostics_dict = {
            "Historical_Nobs_Pre_Aggregation": nobs_pre_aggregation,
            "Historical_Nobs_Post_Aggregation": nobs_post_aggregation,
            "Tails_Used_in_t_Test": hypothesis,
            "Psi_Assignment_Tatio": assignment_ratio,
            "Alpha_Z_Statistic": alpha_z,
            "Beta_Z_Statistic": beta_z,
            "Sample_Stdev": sample_stdev,
            "Degrees_of_Freedom": dof,
            "Theta": theta,
            "Numerator_Y_Bar": y_bar,
            "Denominator_W_Bar": w_bar,
            "Y_Bar_Residuals": y_residuals,
            "W_Bar_Residuals": w_residuals,
            "Estimated_MDE": mde_est,
            "Estimated_Required_Sample_Size": n_est,
            "Estimated_Power": power_est,
        }
        return diagnostics_dict

    _validate_number_of_inputs()
    _validate_input_values()
    if verbose:
        print(
            "Performing power calculation for OLS using a "
            + f"historic dataset of {humanize.intcomma(len(data))} "
            + "observations."
        )
    nobs_pre_aggregation_val = len(data)
    working_df = _aggregate_data(data, endog, exog, cluster, numerator, denominator)
    nobs_post_aggregation_val = len(working_df)
    _validate_sample_size(working_df, exog)
    (
        stdev,
        degrees,
        theta_val,
        y_bar_val,
        w_bar_val,
        y_resid,
        w_resid,
    ) = _compute_sample_stdev(is_ratio, working_df, endog, exog)
    mean_nobs_per_cluster = working_df["NOBS_IN_CLUSTER"].mean()
    if mde is None and n is not None and power is not None:
        mde, alpha_z_stat, beta_z_stat = _derive_mde(
            is_ratio_metric=is_ratio,
            sample_size=n,
            historical_stdev=stdev,
            alpha_value=alpha,
            power_value=power,
            assignment_ratio=ratio,
            w_bar=mean_nobs_per_cluster,
            num_sides=alternative,
            covariates=exog,
        )
        if ratio is None and verbose:
            print(
                "The assignment ratio was not provided. Assuming a 0.5 ratio"
                + "between treated versus control units of asignment."
            )
        diagnostics_dict = _create_diagnostics_dict(
            sample_stdev=stdev,
            dof=degrees,
            theta=theta_val,
            assignment_ratio=ratio,
            y_bar=y_bar_val,
            w_bar=w_bar_val,
            y_residuals=y_resid,
            w_residuals=w_resid,
            hypothesis=alternative,
            nobs_pre_aggregation=nobs_pre_aggregation_val,
            nobs_post_aggregation=nobs_post_aggregation_val,
            alpha_z=alpha_z_stat,
            beta_z=beta_z_stat,
            mde_est=mde,
            n_est=n,
            power_est=power,
        )
        return mde, diagnostics_dict
    if n is None and mde is not None and power is not None:
        n, alpha_z_stat, beta_z_stat = _derive_n(
            is_ratio_metric=is_ratio,
            assumed_mde=mde,
            historical_stdev=stdev,
            alpha_value=alpha,
            power_value=power,
            assignment_ratio=ratio,
            w_bar=mean_nobs_per_cluster,
            num_sides=alternative,
            covariates=exog,
        )
        if ratio is None:
            print(
                "The assignment ratio was not provided. Assuming a 0.5 ratio"
                + "between treated versus control units of asignment."
            )
        diagnostics_dict = _create_diagnostics_dict(
            sample_stdev=stdev,
            dof=degrees,
            theta=theta_val,
            assignment_ratio=ratio,
            y_bar=y_bar_val,
            w_bar=w_bar_val,
            y_residuals=y_resid,
            w_residuals=w_resid,
            hypothesis=alternative,
            nobs_pre_aggregation=nobs_pre_aggregation_val,
            nobs_post_aggregation=nobs_post_aggregation_val,
            alpha_z=alpha_z_stat,
            beta_z=beta_z_stat,
            mde_est=mde,
            n_est=n,
            power_est=power,
        )
        return n, diagnostics_dict
    if power is None and mde is not None and n is not None:
        power, alpha_z_stat = _derive_power(
            is_ratio_metric=is_ratio,
            sample_size=n,
            assumed_mde=mde,
            historical_stdev=stdev,
            alpha_value=alpha,
            assignment_ratio=ratio,
            w_bar=mean_nobs_per_cluster,
            num_sides=alternative,
            covariates=exog,
        )
        if ratio is None and verbose:
            print(
                "The assignment ratio was not provided. Assuming a 0.5 ratio"
                + "between treated versus control units of asignment."
            )
        diagnostics_dict = _create_diagnostics_dict(
            sample_stdev=stdev,
            dof=degrees,
            theta=theta_val,
            assignment_ratio=ratio,
            y_bar=y_bar_val,
            w_bar=w_bar_val,
            y_residuals=y_resid,
            w_residuals=w_resid,
            hypothesis=alternative,
            nobs_pre_aggregation=nobs_pre_aggregation_val,
            nobs_post_aggregation=nobs_post_aggregation_val,
            alpha_z=alpha_z_stat,
            mde_est=mde,
            n_est=n,
            power_est=power,
        )
        return power, diagnostics_dict
