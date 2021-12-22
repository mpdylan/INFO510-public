# INFO510-public

This repository contains notebooks and slides related to INFO 510 / ISTA 410 at the University of Arizona.

## Contents

The `slides` directory contains old slides, from each of the three semesters that I taught it.

The `notebooks` directory contains a number of example notebooks, covering a variety of topics and techniques. Some of the notebooks were written under old versions of PyMC3 and ArviZ and will not run under the current versions -- I'm currently working on fixing them.

Examples are drawn from a variety of sources, including the exercises and examples from Gelman et al's book *Bayesian Data Analysis* (below, BDA3), Richard McElreath's book *Statistical Rethinking* (Rethinking), and the repository owner's (Dylan Murphy) own work.

### Example notebooks

The following list is currently incomplete.

* **Binomial models**. In `binom_models`:
    * `vaccines.ipynb` contains a replication of Pfizer's analysis for the Emergency Use Authorization of their COVID-19 vaccine.
    * `bikelanes.ipynb` contains a hierarchical binomial model for bicycle traffic, following exercises 3.8 and 5.13 from BDA3.
* **Normal models**. In `normal_models`:
    * `basketball.ipynb` contains a normal model for basketball scores.
    * `basketball_with_mcmc.ipynb` reworks the normal model in PyMC3 with MCMC sampling for inference.
    * `eight_schools.ipynb` contains an implementation of the 8 schools model from chapter 5 of BDA3.
* **Linear models**. In `linear_models`:
    * `africa_ruggedness.ipynb` contains a PyMC3 translation of the model describing the relationship between terrain ruggedness and GDP in African and non-African nations from Rethinking. The code in this notebook is derived partly from the translation developed by the PyMC3 development team.
    * `beijing_hlm.ipynb` contains a hierarchical linear model describing ozone levels and temperature at 12 monitoring sites in Beijing, including varying-intercepts and varying-slopes models.
    * `covariance_ucb.ipynb` replicates the analysis of the UC Berkeley graduate admissions data from Rethinking.
    * `opening_blocking.ipynb` has a synthetic-data simulation to illustrate basic principles of opening and closing paths in a DAG for causal inference.
    * `simple_linear_reg.ipynb` is a basic demonstration of linear regression in PyMC3 using the relationship between global temperature and atmospheric carbon dioxide as an example.
    * `wines.ipynb` replicates the analysis of the Judgement of Princeton wine-tasting event from the exercises of McElreath's Statistical Rethinking course.
* **Poisson regression**. In `poisson_models`:
    * `salamanders.ipynb` contains an analysis of salamander counts using Poisson regression, following an exercise from Rethinking.
* **Gaussian process regression**. In `gaussian_processes`:
    * `gp_cherry.ipynb` contains a Gaussian process model for the peak flowering date of cherry blossoms in Kyoto, Japan, using data from the 9th century CE to the present.
    * `gp_bikeshare.ipynb` contains a Poisson regression model for the count of bicycles checked out from the Capital Bikeshare system in Washington, DC over a two year period. In this notebook, we begin with a simple Poisson GLM with temperature and windspeed as predictors, and augment it with a latent Gaussian process to model changes in the popularity of the bike share system over time.
* **Model checking**. In `checking`:
    * `speed_of_light_ppc.ipynb` contains an example of using posterior predictive checking to assess the performance of a simple normal model (following BDA chapter 2)
    * `waic_overfitting_working.ipynb` contains calculations using synthetic data to demonstrate AIC as an estimate of out-of-sample deviance.
* **Sampling methods implemented by hand**: In `sampling`, two notebooks implementing Metropolis and grid-approximation inverse CDF sampling from scratch.
* **Time series models**. In `time_series`: notebooks demonstrating simple implementations of Kalman filters and hidden Markov models (not using PyMC3).
* **Miscellaneous models**. `kidney_cancer`, `election`, and `election_forecast` each attempt to reproduce the results of some examples from BDA3.
