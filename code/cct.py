import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt


def load_data():
    # loading data from other directory
    data = np.array(pd.read_csv('/home/jovyan/cct-midterm/data/plant_knowledge.csv'))

    # data cleaning
    data = data[:, 1:]
    data = data.astype(int)
    return data

def cct(data):
    # N is rows in data (participants), M is columns (questions)
    N, M = data.shape

    with pm.Model() as model:
        # priors
        D = pm.Uniform("D", lower=0.5, upper=1, shape=N)
        Z = pm.Bernoulli("Z", p=0.5, shape=M)
        D_reshaped = D[:, None]

        # calculate p
        p = Z * D_reshaped + (1 - Z) * (1 - D_reshaped)

        # likelihood matrix
        X_obs = pm.Bernoulli("X_obs", p=p, observed=data)

        # sample from post, target_accept = 0.95 for more accurate model
        trace = pm.sample(draws=2000, chains=4, tune=1000, target_accept=0.95)

    # competence estimation per informant
    summary_D = az.summary(trace, var_names=["D"])
    posterior_mean_D = summary_D["mean"]
    
    # reindex informants to better reflect data
    posterior_mean_D.index = np.arange(1, N+1)
    print("Posterior Mean Competence for Each Informant:")
    print(posterior_mean_D.to_string())

    print('Most competent: Informant #' + str(posterior_mean_D.values.argmax() + 1))
    print('Least competent: Informant #' + str(posterior_mean_D.values.argmin() + 1))

    # visualize & save competence
    az.plot_posterior(trace, var_names=["D"])
    plt.savefig("competence_posterior.png")
    print("Plot saved as 'competence_posterior.png'")

    # posterior consensus
    summary_Z = az.summary(trace, var_names=["Z"])
    posterior_mean_Z = summary_Z["mean"]
    
    # reindex
    posterior_mean_Z.index = np.arange(1, M+1)
    print("Posterior Mean Probability for Each Consensus Answer:")
    print(posterior_mean_Z.to_string())

    # consensus answer key
    most_likely_Z = posterior_mean_Z.round().astype(int)
    print("Consensus answer key:")
    print(most_likely_Z.to_string())

    # visualize & save consensus
    az.plot_posterior(trace, var_names=["Z"])
    plt.savefig("consensus_posterior.png")
    print("Plot saved as 'consensus_posterior.png'")

    # majority vote comparison
    majority_vote = (data.sum(axis=0) >= (N / 2)).astype(int)
    maj_series = pd.Series(majority_vote, index=np.arange(1, M+1))
    print("Majority answer key:")
    print(maj_series.to_string())

    diffs = np.where(majority_vote != most_likely_Z.values)[0] + 1
    print(f"Differences at questions: {diffs.tolist()}")

#main
data = load_data()
cct(data)
