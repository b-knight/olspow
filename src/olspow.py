import humanize
import math
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm


NOBS_PER_COVARIATE = 100


def solve_power(
    data,
    endog,
    exog,
    cluster,
    n=None,
    mde=None,
    power=None,
    ratio=0.5,
    alpha=0.05,
    alternative="two-sided",
    verbose=False,
):
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

    def _validate_input_values():
        """
        Validate the input values for the OLS Power analysis.

        Parameters:
        - data (pd.DataFrame): The input data as a pandas DataFrame.
        - endog (str): The name of the response variable.
        - exog (list): The names of the predictor variables.
        - cluster (str): The name of the entity to receive experimental assignment.
        - n (int): The number of observations to be used in the analysis.
        - alpha (float): The significance level of the test.
        - power (float): The desired level of statistical power.
        - ratio (float): The proportion of assignment entities to be treated.
        - alternative (string): Whether to performa a one-sided or two-sided test.
        Raises:
        - ValueError: If any of the input values are invalid.

        Returns:
        - None
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("The 'data' argument must be a pandas DataFrame.")
        if not isinstance(endog, str):
            raise ValueError(
                "The 'endog' argument must be a string corresponding to the "
                + "name of the response variable."
            )
        if endog.lower() not in [x.lower() for x in data.columns]:
            raise ValueError(
                f"The response variable '{endog}' was not found within "
                + "the provided Pandas dataframe."
            )
        if not isinstance(exog, list):
            raise ValueError(
                "The 'exog' argument must be a list of strings corresponding "
                + "to the names of the predictor variables."
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
        if not isinstance(cluster, str):
            raise ValueError(
                "The 'cluster' argument must be a string corresponding to the "
                + "name of the entity to receive experimental assignment."
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
                + "corresponding to the type of hypothesis test to be performed."
            )

    def _create_data_shape(data, endog, exog, cluster, agg_method=None):
        """
        Create a shaped dataset by aggregating the data based on
        specified variables.

        Parameters:
            data (pandas.DataFrame): The input data.
            endog (str): The endogenous variable.
            exog (list): The exogenous variables.
            cluster (str): The clustering variable.
            agg_method (dict, optional): The aggregation method for
            each variable.

        Returns:
            pandas.DataFrame: The aggregated dataset.

        """
        if verbose:
            print(f"Aggregating data on the cluster key: {cluster}")
        if agg_method is None:
            agg_method = {}
            agg_method[endog], agg_method[cluster] = (
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

    def _validate_post_agg_data_shape(data, exog):
        """
        Validates the shape of the post-aggregation data.

        Args:
            data (pandas.DataFrame): The post-aggregation data.

        Returns:
            None
        """
        num_coef = len(exog) + 2
        if len(data) / (num_coef) < NOBS_PER_COVARIATE:
            print(
                f"NOTE: Model specification implies {num_coef} covariates "
                + f"({num_coef} coefficient estimates). \nHowever, only "
                + f"{humanize.intcomma(len(data))} unique units of assignment "
                + "were found. \nConsider increasing the size of the "
                + "historical dataset to at least "
                + f"{humanize.intcomma(NOBS_PER_COVARIATE)} "
                + "units of assignment per coefficient estimate."
            )
        if len(data) < num_coef:
            raise ValueError(
                f"The number of units of assignment in the historical dataset "
                + f"({len(data)}) is insufficient to estimate the model. "
                + f"\nTry increasing the size of the historical dataset to at "
                + f"least {humanize.intcomma(num_coef*100)} units "
                + "of assignment."
            )

    def _compute_residuals_of_response_var(data, endog, exog):
        """
        Compute the residuals of the response variable.

        Args:
            data (pandas.DataFrame): The input data.
            endog (str): The endogenous variable.
            exog (list): The exogenous variables.

        Returns:
            pandas.DataFrame: The residuals of the response variable.

        """
        if verbose:
            print("Computing the residuals of the response variable.")
        X, Y = data[exog], data[endog]
        X = sm.add_constant(X)
        model = sm.OLS(Y, X)
        model_fit = model.fit()
        return model_fit.resid

    def _compute_residuals_of_nobs_per_cluster(data, exog):
        """
        Compute the residuals corresponding to the number of observations per cluster.

        Args:
            data (pandas.DataFrame): The input data containing the exogenous
            variables and the number of observations per cluster.
            exog (str or list): The name(s) of the exogenous variable(s) in the data.

        Returns:
            pandas.Series: The residuals of the OLS model fit.

        """
        if verbose:
            print(
                "Computing the residuals corresponding to the "
                + "number of observation per cluster."
            )
        X, Y = data[exog], data["NOBS_IN_CLUSTER"]
        X = sm.add_constant(X)
        model = sm.OLS(Y, X)
        model_fit = model.fit()
        return model_fit.resid

    def _compute_ratio_of_means(working_df, endog):
        """
        Computes the ratio_of_means (mean of x / mean of y)

        Parameters:
        working_df (pandas.DataFrame): The working dataframe.
        endog (str): The name of the endogenous variable.

        Returns:
        float: The computed ratio_of_means value.
        """
        return working_df[endog].mean() / working_df["NOBS_IN_CLUSTER"].mean()

    def _computer_standard_deviation(working_df, endog, exog, dof):
        """
        Compute the standard deviation of the double residuals.

        Parameters:
        - working_df (pandas.DataFrame): The working dataframe.
        - endog (str): The name of the response variable.
        - exog (list): The names of the explanatory variables.
        - dof (int): The degrees of freedom.

        Returns:
        - float: The standard deviation of the double residuals.
        """
        j_resid = _compute_residuals_of_response_var(working_df, endog, exog)
        n_resid = _compute_residuals_of_nobs_per_cluster(working_df, exog)
        resid_data = pd.DataFrame([j_resid, n_resid]).T
        resid_data.columns = ["j_resid", "n_resid"]
        resid_data["ratio_of_means"] = _compute_ratio_of_means(working_df, endog)
        resid_data["double_residual"] = resid_data["j_resid"] - (
            resid_data["ratio_of_means"] * resid_data["n_resid"]
        )
        return np.sqrt(((resid_data["double_residual"] ** 2).sum()) / dof)

    def _fetch_z_statistic(input, alternative):
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
        Derives the minimum detectable effect (MDE) based on the given parameters.

        Args:
            sample_size (int): The sample size.
            historical_stdev (float): The historical standard deviation.
            w_bar (float): The average treatment effect.
            assignment_ratio (float): The assignment ratio.
            alpha_value (float): The significance level.
            power_value (float): The desired power.
            num_sides (int): The number of sides for the hypothesis test.
            covariates (list): The list of covariates.

        Returns:
            float: The minimum detectable effect (MDE).

        """
        if verbose:
            print("Deriving the minimum detectable effect.")
        alpha_z, alpha = _fetch_z_statistic(alpha_value, num_sides)
        beta_z, beta = _fetch_z_statistic(1.0 - power_value, num_sides)
        mde = (
            (alpha_z + beta_z)
            * np.sqrt((1 / assignment_ratio) + 1 / (1 - assignment_ratio))
            * (historical_stdev / (w_bar * np.sqrt(sample_size)))
        )
        if verbose:
            print("-" * 80)
        print(
            f"The minimum detectable effect (MDE) was estimated as {humanize.intcomma(mde)}. "
            + f"\nEstimate was based on {humanize.intcomma(len(data))} observations "
            + f"across {humanize.intcomma(len(working_df))} clusters of historical data. "
            + f"\nThe model specification used {humanize.intcomma(len(covariates))} "
            + f"covariates, assumes {humanize.intcomma(len(covariates)+1)} coefficient "
            + f"estimates, and {humanize.intcomma(len(working_df) - len(covariates)+1)} "
            + f"degrees of freedom (alpha={round(alpha, 4)}, "
            + f"beta={round(beta, 4)})."
        )
        return mde

    def _derive_n(
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
        Derives the required sample size for a power analysis.

        Args:
            assumed_mde (float): The assumed minimum detectable effect (MDE).
            historical_stdev (float): The standard deviation of the historical data.
            w_bar (float): The average treatment effect.
            assignment_ratio (float): The ratio of treatment group size to control group size.
            alpha_value (float): The significance level (alpha).
            power_value (float): The desired power value (1 - beta).
            num_sides (int): The number of sides for the hypothesis test (1 or 2).
            covariates (list): The list of covariates used in the model.

        Returns:
            int: The minimum required sample size.

        """

        if verbose:
            print("Deriving the required sample size.")
        alpha_z, alpha = _fetch_z_statistic(alpha_value, num_sides)
        beta_z, beta = _fetch_z_statistic(1.0 - power_value, num_sides)
        sample_size = math.ceil(
            (
                ((alpha_z + beta_z) ** 2)
                * ((1 / assignment_ratio) + 1 / (1 - assignment_ratio))
                * ((historical_stdev) ** 2 / ((w_bar**2) * (assumed_mde**2)))
            )
        )
        if verbose:
            print("-" * 80)
        print(
            "The minimum required sample size was estimated as "
            + f"{humanize.intcomma(sample_size)} using a minimum "
            + f"detectable effect (MDE) of {humanize.intcomma(assumed_mde)}. "
            + f"\nEstimate was based on {humanize.intcomma(len(data))} observations "
            + f"across {humanize.intcomma(len(working_df))} clusters of historical data. "
            + f"\nThe model specification used {humanize.intcomma(len(covariates))} "
            + f"covariates, assumes {humanize.intcomma(len(covariates)+1)} coefficient "
            + f"estimates, and {humanize.intcomma(len(working_df) - len(covariates)+1)} "
            + f"degrees of freedom (alpha={round(alpha, 4)}, "
            + f"beta={round(beta, 4)})."
        )
        return sample_size

    def _derive_power(
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
        Calculates the estimated statistical power.

        Parameters:
        - sample_size (int): The sample size.
        - assumed_mde (float): The assumed minimum detectable effect (MDE).
        - historical_stdev (float): The historical standard deviation.
        - alpha_value (float): The alpha value.
        - assignment_ratio (float): The assignment ratio.
        - w_bar (float): The w_bar value.
        - num_sides (int): The number of sides.
        - covariates (list): The list of covariates.

        Returns:
        - est_power (float): The estimated statistical power.
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
        print(
            "Statistical power was estimated as "
            + f"{round(est_power, 4)} using a minimum "
            + f"detectable effect (MDE) of {humanize.intcomma(assumed_mde)}. "
            + f"\nEstimate was based on {humanize.intcomma(len(data))} observations "
            + f"across {humanize.intcomma(len(working_df))} clusters of historical data. "
            + f"\nThe model specification used {humanize.intcomma(len(covariates))} "
            + f"covariates, assumes {humanize.intcomma(len(covariates)+1)} coefficient "
            + f"estimates, and {humanize.intcomma(len(working_df) - len(covariates)+1)} "
            + f"degrees of freedom (alpha={round(alpha, 4)})."
        )
        return est_power

    _validate_number_of_inputs()
    _validate_input_values()
    if verbose:
        print(
            "Performing power calculation for OLS using a "
            + f"historic dataset of {humanize.intcomma(len(data))} "
            + "observations."
        )
    working_df = _create_data_shape(data, endog, exog, cluster)
    _validate_post_agg_data_shape(working_df, exog)
    dof = len(working_df) - (len(exog) + 1)
    stdev = _computer_standard_deviation(working_df, endog, exog, dof)
    mean_nobs_per_cluster = working_df["NOBS_IN_CLUSTER"].mean()
    if mde is None and n is not None and power is not None:
        mde = _derive_mde(
            sample_size=n,
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
        return mde
    if n is None and mde is not None and power is not None:
        n = _derive_n(
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
        return n
    if power is None and mde is not None and n is not None:
        power = _derive_power(
            sample_size=n,
            assumed_mde=mde,
            historical_stdev=stdev,
            alpha_value=alpha,
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
        return power
