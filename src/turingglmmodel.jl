
"""
    TuringGLMModel{T}

Bayesian regression model container that holds the formula, data, priors, and fitted samples.
"""
mutable struct TuringGLMModel{T<:UnivariateDistribution}
    formula::FormulaTerm
    model::Model
    prior::CustomPrior
    link::Function
    y::AbstractVector
    X::AbstractMatrix
    Z::Union{Nothing,Matrix}
    μ_X::AbstractVector
    σ_X::AbstractVector
    μ_y::AbstractFloat
    σ_y::AbstractFloat
    X_names::Tuple
    Z_names::Tuple
    standardized::Bool
    samples::Union{Nothing,Chains}
    unstd_params::Union{Nothing, Chains}
end

# Constructor with names, used in _extended_turing_model()
function TuringGLMModel(
    ::Type{T},
    formula::FormulaTerm,
    model::Model,
    prior::CustomPrior,
    y::AbstractVector,
    X::AbstractMatrix,
    Z::Union{Nothing,Matrix},
    μ_X::AbstractVector,
    σ_X::AbstractVector,
    μ_y::AbstractFloat,
    σ_y::AbstractFloat,
    standardized::Bool,
) where {T<:UnivariateDistribution}

    init_X_names = Symbol.(formula.rhs)
    init_Z_names = () #No Z supported
    link = get_link(T)

    return TuringGLMModel{T}(
        formula,
        model,
        prior,
        link,
        y,
        X,
        Z,
        μ_X,
        σ_X,
        μ_y,
        σ_y,
        init_X_names,
        init_Z_names,
        standardized,
        nothing,
        nothing
    )
end

function Base.show(io::IO, TM::TuringGLMModel{T}; warnings=true) where {T}
    # Define styles once at the top
    header_style = crayon"bold underline"
    label_style = crayon"bold !underline"
    normal_style = crayon"reset"  # or just use no crayon

    println(io, header_style, "TuringGLM Model")

    # Family
    print(io, label_style, "Family: ")
    family_string = "$T (link: $(string(TM.link)))"
    println(io, normal_style, family_string)

    # Formula  
    print(io, label_style, "Formula: ")
    println(io, normal_style, string(TM.formula))

    # Prior
    println(io, label_style, "Prior:")
    pr = TM.prior
    print(io, normal_style, "  Predictors: ")
    println(io, normal_style, clean_prior_string(string(pr.predictors)))
    print(io, normal_style, "  Intercept: ")
    println(io, normal_style, clean_prior_string(string(pr.intercept)))

    if !isnothing(pr.auxiliary)
        print(io, normal_style, "  Auxiliary: ")
        println(io, normal_style, clean_prior_string(string(pr.auxiliary)))
    end

    # Observations
    print(io, label_style, "Observations: ")
    println(io, normal_style, size(TM.X, 1))

    # Samples
    print(io, label_style, "Samples: ")
    if isnothing(TM.samples)
        println(io, normal_style, "empty")
    else
        sz = size(TM.samples)
        println(io, normal_style, "$(sz[1] * sz[3]) samples across $(sz[3]) chains")
    end

    if warnings
        println(io)
        model_warnings(TM)
    end
end

#### Methods ####

"""
    fit!(TM::TuringGLMModel; sampler=NUTS(), parallel=MCMCThreads(), N=2000, nchains=4, quiet=true, kwargs...)

Fit the model using MCMC sampling. Updates the model in-place with results.
    Kwargs are passed to Turing's `sample()` function.

# Arguments
- `sampler`: MCMC sampler (default: NUTS())
- `parallel`: Parallelization method (default: MCMCThreads())
- `N`: Number of samples per chain
- `nchains`: Number of parallel chains
- `quiet`: Suppress sampling output
"""
function fit!(
    TM::TuringGLMModel;
    sampler=NUTS(),
    parallel=MCMCThreads(),
    N=2000,
    nchains=4,
    quiet=true,
    kwargs...,
)
    if quiet
        TM.samples = @suppress sample(TM.model, sampler, parallel, N, nchains; kwargs...)
    else
        TM.samples = sample(TM.model, sampler, parallel, N, nchains; kwargs...)
    end
    _make_unstd_parameters!(TM)
    return TM
end
