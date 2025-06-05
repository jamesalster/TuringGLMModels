
# TuringArm.jl

A simple wrapper around [TuringGLM.jl](https://turinglang.org/TuringGLM.jl/stable/) for Bayesian regression.

## Installation

```julia
using Pkg
Pkg.add("TuringArm")
```

## Usage

```julia
using TuringArm, RDatasets, Statistics

# Load car data
mtcars = dataset("datasets", "mtcars")

# Create a model
mod = turing_glm(
    @formula(MPG ~ Cyl + Disp),
    mtcars,
    Normal
)

# Fit the model
fit!(mod, N=1000, nchains=2)

# View results
pretty(mod)

# Get coefficients
fixef(mod)  # With uncertainty
coefs(mod)  # Point estimates, equivalent to fixef(mod,  median)
fixef(mod, std) # Reduce to point estimate with passed function
fixef(mod, x -> quantile(x, [0.05, 0.95])) # Reduce with custom function

# Extract parameters with options
parameters(mod, drop_warmup=100, n_draws=500, collapse=false)
internals(mod, median)  # Get variance parameters

# Make predictions
predict(mod)  # For original data
predict(mod, type=:epred)  # Expected values
predict(mod, type=:linpred)  # Linear predictor

# Predict on new data
new_data = [6.0 200.0; 8.0 350.0]
predict(mod, new_data, mean) # Optionally pass fucntion

# Compare models
robust_mod = turing_glm(@formula(MPG ~ Cyl + Disp), mtcars, TDist)
fit!(robust_mod, N=1000, nchains=2)

loo_compare(mod, robust_mod)
```

## API

### Model Creation
* `turing_glm(formula, data, family)` - Create from formula and data
* `turing_glm(y, X, family)` - Create from matrices

### Fitting
* `fit!(model)` - Run MCMC sampling

### Parameter Extraction
* `parameters(model, fun)` - All parameters
* `fixef(model, fun)` - Fixed effects  
* `internals(model, fun)` - Sampling information 
* `coefs(model, fun)` - Point estimates
* `get_parameters(model, params)` - Specific parameters

### Predictions
* `predict(model, X)` - Generate predictions
* `linpred(model, X)` - Linear predictor
* `epred(model, X)` - Expected values
* `posterior_pred(model, X)` - Posterior predictive samples

### Model Comparison
* `psis_loo(model)` - Leave-one-out cross-validation
* `loo_compare(models...)` - Compare multiple models

### Utilities
* `pretty(model)` - Formatted summary
* `parameter_names(model)` - Parameter names
* `outcome(model)` - Response variable

### Common Arguments

Parameter extraction functions accept:

* `drop_warmup=200` - Warmup samples to drop
* `n_draws=-1` - Number of draws (-1 for all)
* `collapse=true` - Collapse chains into single dimension

## Thanks

This package simply wraps [TuringGLM.jl](https://turinglang.org/TuringGLM.jl/stable/) with a streamlined interface. Turing and TuringGLM do all the heavy lifting.

## TODO

* Revise readme and docs so not written by LLM
* Add testing
* Add simple plots
* Add support for random effects
