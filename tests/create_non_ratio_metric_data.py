import itertools
import numpy as np
import pandas as pd
from scipy.stats import norm

np.random.seed(7)  # Set a seed for reproducibility


def gen_data(
    n_c=5,  # number of clusters
    mu_nobs_per_c=1,  # mean number of nobs per cluster
    std_nobs_per_c=1,  # stddev of nobs per cluster
    psi=0.5,  # proportion of clusters treated
    epsilon_mu=0,  # the mean amount of noise to be added to an observation
    epsilons_std=5,  # the stddev of the noise to be added to an observation
    # -------------------------------------------------------------------------------------------------
    mu_of_mu_c_treated=5,  # the mean of the cluster means for the treatment effect
    std_of_mu_c_treated=1,  # stddev of cluster means for treatment effect
    mu_of_c_std_treated=1,  # the mean of the cluster's stddev for the treatment effect
    std_of_c_std_treated=1,  # the stddev of the cluster's stddev for the treatment effect
    # -------------------------------------------------------------------------------------------------
    mu_of_mu_c_cov_1=120,  # the mean of the cluster means for covariate 1
    std_of_mu_c_cov_1=20,  # stddev of cluster means for covariate 1
    mu_of_c_std_cov_1=10,  # the mean of the cluster's stddev for covariate 1
    std_of_c_std_cov_1=5,  # the stddev of the cluster's stddev for covariate 1
    # -------------------------------------------------------------------------------------------------
    mu_of_mu_c_cov_1_eff=0.5,  # the mean of the cluster means for the coef est. of covariate 1
    std_of_mu_c_cov_1_eff=0.06,  # stddev of cluster means for the coef est. of covariate 1
    mu_of_c_std_cov_1_eff=0.01,  # the mean of the cluster's stddev for the coef est. of covariate 1
    std_of_c_std_cov_1_eff=0.01,  # the stddev of the cluster's stddev for the coef est. of covariate 1
    # -------------------------------------------------------------------------------------------------
    mu_of_mu_c_cov_2=20,  # the mean of the cluster means for covariate 2
    std_of_mu_c_cov_2=5,  # stddev of cluster means for covariate 2
    mu_of_c_std_cov_2=2,  # the mean of the cluster's stddev for covariate 2
    std_of_c_std_cov_2=1,  # the stddev of the cluster's stddev for covariate 2
    # -------------------------------------------------------------------------------------------------
    mu_of_mu_c_cov_2_eff=1.5,  # the mean of the cluster means for the coef est. of covariate 2
    std_of_mu_c_cov_2_eff=0.5,  # stddev of cluster means for the coef est. of covariate 2
    mu_of_c_std_cov_2_eff=0.25,  # the mean of the cluster's stddev for the coef est. of covariate 2
    std_of_c_std_cov_2_eff=0.1,  # the stddev of the cluster's stddev for the coef
):
    nob_ids, cluster_ids, received_treatment, nobs_in_cluster = [], [], [], []
    treatment_values, covariate_1_values, covariate_1_coefs = [], [], []
    covariate_2_values, covariate_2_coefs, epsilon_values = [], [], []
    nob_id = 0
    cluster_id = 0

    nobs_per_cluster = norm.ppf(
        np.random.random(n_c),
        loc=mu_nobs_per_c,
        scale=std_nobs_per_c**2,
    ).astype(int)

    for assignment_entity in nobs_per_cluster:
        # treatment_values
        if np.random.uniform(low=0, high=1) <= psi:
            for nob in range(0, max(assignment_entity, 1)):
                received_treatment.append(1)
            cluster_mean_treatment_val = np.random.normal(
                loc=mu_of_mu_c_treated, scale=std_of_mu_c_treated
            )
            cluster_std_treatment_val = abs(
                np.random.normal(loc=mu_of_c_std_treated, scale=std_of_c_std_treated)
            )
            treatment_values.append(
                list(
                    np.random.normal(
                        loc=cluster_mean_treatment_val,
                        scale=cluster_std_treatment_val,
                        size=max(assignment_entity, 1),
                    )
                )
            )
        else:
            for nob in range(0, max(assignment_entity, 1)):
                received_treatment.append(0)
                treatment_values.append([0])
        for nob in range(0, max(assignment_entity, 1)):
            nob_ids.append(nob_id)
            nob_id += 1
            cluster_ids.append(cluster_id)
            nobs_in_cluster.append(max(assignment_entity, 1))
        cluster_id += 1

        # covariate 1
        cluster_mean_cov_1_val = np.random.normal(
            loc=mu_of_mu_c_cov_1, scale=std_of_mu_c_cov_1
        )
        cluster_std_cov_1_val = abs(
            np.random.normal(loc=mu_of_c_std_cov_1, scale=std_of_c_std_cov_1)
        )
        covariate_1_values.append(
            list(
                np.random.normal(
                    loc=cluster_mean_cov_1_val,
                    scale=cluster_std_cov_1_val,
                    size=max(assignment_entity, 1),
                )
            )
        )

        cluster_mean_cov_coef = np.random.normal(
            loc=mu_of_mu_c_cov_1_eff, scale=std_of_mu_c_cov_1_eff
        )
        cluster_std_cov_1_coef = abs(
            np.random.normal(loc=mu_of_c_std_cov_1_eff, scale=std_of_c_std_cov_1_eff)
        )
        covariate_1_coefs.append(
            list(
                np.random.normal(
                    loc=cluster_mean_cov_coef,
                    scale=cluster_std_cov_1_coef,
                    size=max(assignment_entity, 1),
                )
            )
        )

        # covariate 2
        cluster_mean_cov_2_val = np.random.normal(
            loc=mu_of_mu_c_cov_2, scale=std_of_mu_c_cov_2
        )
        cluster_std_cov_2_val = abs(
            np.random.normal(loc=mu_of_c_std_cov_2, scale=std_of_c_std_cov_2)
        )
        covariate_2_values.append(
            list(
                np.random.normal(
                    loc=cluster_mean_cov_2_val,
                    scale=cluster_std_cov_2_val,
                    size=max(assignment_entity, 1),
                )
            )
        )

        cluster_mean_cov_coef = np.random.normal(
            loc=mu_of_mu_c_cov_2_eff, scale=std_of_mu_c_cov_2_eff
        )
        cluster_std_cov_2_coef = abs(
            np.random.normal(loc=mu_of_c_std_cov_2_eff, scale=std_of_c_std_cov_2_eff)
        )
        covariate_2_coefs.append(
            list(
                np.random.normal(
                    loc=cluster_mean_cov_coef,
                    scale=cluster_std_cov_2_coef,
                    size=max(assignment_entity, 1),
                )
            )
        )

    # add noise
    epsilon_values = np.random.normal(
        loc=epsilon_mu, scale=epsilons_std, size=len(nob_ids)
    )

    data = pd.DataFrame(
        {
            "ID": nob_ids,
            "CLUSTER_ID": cluster_ids,
            "TREATED_CLUSTER": received_treatment,
            "NOBS_IN_CLUSTER": nobs_in_cluster,
            "TREATMENT_EFFECT": list(itertools.chain.from_iterable(treatment_values)),
            "X1_VALUE": list(itertools.chain.from_iterable(covariate_1_values)),
            "X1_COEF": list(itertools.chain.from_iterable(covariate_1_coefs)),
            "X2_VALUE": list(itertools.chain.from_iterable(covariate_2_values)),
            "X2_COEF": list(itertools.chain.from_iterable(covariate_2_coefs)),
            "EPSILON": epsilon_values,
        }
    )
    data["Y"] = (
        data["X1_VALUE"] * data["X1_COEF"]
        + data["X2_VALUE"] * data["X2_COEF"]
        + data["TREATED_CLUSTER"] * data["TREATMENT_EFFECT"]
        + data["EPSILON"]
    )
    data = data[
        [
            "ID",
            "CLUSTER_ID",
            "NOBS_IN_CLUSTER",
            "Y",
            "TREATED_CLUSTER",
            "TREATMENT_EFFECT",
            "X1_VALUE",
            "X1_COEF",
            "X2_VALUE",
            "X2_COEF",
            "EPSILON",
        ]
    ]
    return data
