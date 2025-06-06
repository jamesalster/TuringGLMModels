
"""
    turing_glm(formula::FormulaTerm, data, T; priors=DefaultPrior(), standardize=true)
    turing_glm(y::AbstractVector, X::AbstractArray, T; names=Symbol[], kwargs...)

Create a Bayesian regression model. NB Custom priors should (for now) be specified on a *standardized* (z-scored) model scale.

# Arguments
- `formula`: Model formula specifying response and predictors
- `data`: Dataset containing variables referenced in formula
- `T`: Distribution family (Normal, Bernoulli, Poisson, etc.)
- `priors`: Prior specifications (default: DefaultPrior())
- `standardize`: Whether to standardize predictors and response 
    (response only for families `Normal` or `TDist`). NB TuringGLMs
    implementation may not perform correctly if data is not standardized.

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
    standardize::Bool=true,
) where {T<:UnivariateDistribution}

    # warning for not standardized
    standardize ||
        @warn "TuringGLM's model implementation may not perform correctly on unstandardized data."

    # Get what we need
    y = TuringGLM.data_response(formula, data)
    X = TuringGLM.data_fixed_effects(formula, data)
    Z = TuringGLM.data_random_effects(formula, data)

    # Error if random effects
    ranef = TuringGLM.ranef(formula)
    isnothing(ranef) || throw(
        ArgumentError(
            "TuringGLMModels (unlike TuringGLM) does not yet support random effects."
        ),
    )

    # Standardize X
    if standardize
        μ_X, σ_X, X_std = TuringGLM.standardize_predictors(X)
    else
        X_std = X
        μ_X = zeros(size(X, 2))
        σ_X = ones(size(X, 2))
    end

    # Standardize Y
    if standardize && (T ∈ [Normal, TDist])
        μ_y, σ_y, y_std = TuringGLM.standardize_predictors(y)
    else
        y_std = y
        μ_y = 0.0
        σ_y = 1.0
    end

    # Make standardized dataframe
    std_data = DataFrame(hcat(y_std, X_std), Symbol.([formula.lhs, formula.rhs...]))

    #Get prior
    prior = TuringGLM._prior(priors, y_std, T)

    # Make the TuringGLM model
    constructed_model = TuringGLM._turing_model(
        formula, std_data, T; priors, standardize=false
    )

    # Construct TuringArm object
    return TuringGLMModel(
        T,
        formula,
        constructed_model,
        prior,
        y_std,
        X_std,
        Z,
        μ_X,
        σ_X,
        μ_y,
        σ_y,
        standardize,
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
