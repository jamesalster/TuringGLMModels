
"""
    turing_glm(formula::FormulaTerm, data, T; priors=DefaultPrior(), standardize=false)
    turing_glm(y::AbstractVector, X::AbstractArray, T; names=Symbol[], kwargs...)

Create a Bayesian regression model.

# Arguments
- `formula`: Model formula specifying response and predictors
- `data`: Dataset containing variables referenced in formula
- `T`: Distribution family (Normal, Bernoulli, Poisson, etc.)
- `priors`: Prior specifications (default: DefaultPrior())
- `standardize`: Whether to standardize predictors and response

Alternative method:
- `y`: Response vector
- `X`: Predictor matrix
- `names`: Variable names for X columns (auto-generated if empty)
"""
## Main constructor for API
function turing_glm(
    formula::FormulaTerm,
    data,
    ::Type{T};
    priors::Prior=DefaultPrior(),
    standardize::Bool=false,
) where {T<:UnivariateDistribution}
    standardize && @warn "The TuringArm Model object will contain standardized data."
    # Get what we need
    y = TuringGLM.data_response(formula, data)
    X = TuringGLM.data_fixed_effects(formula, data)
    Z = TuringGLM.data_random_effects(formula, data)
    prior = TuringGLM._prior(priors, y, T)
    if standardize
        μ_X, σ_X, X_std = TuringGLM.standardize_predictors(X)
        μ_y, σ_y, y_std = TuringGLM.standardize_predictors(y)
    else
        X_std = X
        y_std = y
    end

    # Make the TuringGLM model
    constructed_model = TuringGLM._turing_model(formula, data, T; priors, standardize)

    # Construct TuringArm object
    return TuringGLMModel(
        T, formula, constructed_model, prior, y_std, X_std, Z, standardize
    )
end

## Method for y and X
function turing_glm(
    y::AbstractVector,
    X::AbstractArray,
    ::Type{T};
    names::Vector{Symbol}=Symbol[],
    kwargs...,
) where {T<:UnivariateDistribution}
    if isempty(names)
        X_names = ntuple(i -> Symbol("X$i"), size(X, 2))
    else
        X_names = ntuple(i -> Symbol(names[i]), length(names))
    end
    table = (; NamedTuple{X_names}(eachcol(X))..., NamedTuple{(:y,)}([y])...)
    #TODO improve
    formula = "y ~ " * join([string.(term) for term in X_names], " + ")
    formula_obj = eval(Meta.parse("@formula($formula)"))
    return turing_glm(formula_obj, table, T; kwargs...)
end
