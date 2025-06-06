"""
    predict(TM::TuringGLMModel, X::AbstractArray, fun=nothing; type=:posterior, transform=TM.standardized, std=false, kwargs...)
    predict(TM::TuringGLMModel, fun=nothing; kwargs...)

Generate predictions for new data or fitted data.

# Arguments
- `X`: Design matrix for predictions (optional, uses fitted data if omitted)
- `fun`: Optional function to apply across draws
- `type`: Type of prediction (:posterior, :epred, :linpred)
- `transform`: Standardize data before feeding to model? 
- `std`: Return predictions at the standardized scale? 
- `drop_warmup`: Number of warmup samples to drop from each chain
- `n_draws`: Number of draws to keep (-1 for all post-warmup)
- `collapse`: Whether to collapse chains into single dimension
- `dropdims`: Whether to drop singleton dimensions (default: true)
"""
function predict(
    TM::TuringGLMModel{T},
    X::AbstractArray,
    fun::Union{Nothing,Function}=nothing;
    type::Symbol=:posterior,
    std::Bool=false,
    transform::Bool=true,
    kwargs...,
) where T
    # Transform
    X_trans = transform ? (X .- TM.μ_X') ./ TM.σ_X' : X

    if type === :posterior
        preds = posterior_pred(TM, X_trans, fun; kwargs...)
    elseif type === :epred
        preds = epred(TM, X_trans, fun; kwargs...)
    elseif type === :linpred
        preds = linpred(TM, X_trans, fun; kwargs...)
    else
        throw(ArgumentError("type must be one of :posterior, :epred or :linpred"))
    end

    # Unstandardize
    if !std && T ∈ [Normal, TDist]
        unstandardize_predictions!(preds, TM)
    end

    return preds
end

function predict(TM::TuringGLMModel, fun::Union{Nothing,Function}=nothing; kwargs...)
    # Do not transform because it is already
    return predict(TM, TM.X, fun; transform=false, kwargs...)
end

#### Internal functions ####
function linpred(
    TM::TuringGLMModel,
    X::AbstractArray,
    fun::Union{Nothing,Function}=nothing;
    dropdims=true,
    kwargs...,
)

    α = get_parameters(TM, [:α]; std=true, kwargs...) # vec required for NamedArray problems below
    β = fixef(TM; std=true, kwargs...)
    # handle 3d
    μ = zeros(
        eltype(α), (Dim{:row}(size(X, 1)), Dim{:draw}(size(α, 1)), Dim{:chain}(size(α, 3)))
    )
    for i in 1:size(μ, 3)
        μ[:, :, i] = vec(α[:, :, i])' .+ X * β[:, :, i]'
    end
    μ = isnothing(fun) ? μ : mapslices(fun, μ; dims=2)
    return dropdims ? drop_single_dims(μ) : μ
end

function epred(
    TM::TuringGLMModel{T},
    X::AbstractArray,
    fun::Union{Nothing,Function}=nothing;
    dropdims=true,
    kwargs...,
) where {T}
    μ = linpred(TM, X; kwargs...) # don't pass fun
    invlink = let
        if TM.link == identity
            identity
        elseif TM.link == logit
            logistic
        elseif TM.link == log
            exp
        end
    end
    epreds = invlink.(μ)
    epreds = isnothing(fun) ? epreds : mapslices(fun, epreds; dims=2)
    return dropdims ? drop_single_dims(epreds) : epreds
end

function posterior_pred(
    TM::TuringGLMModel{T},
    X::AbstractArray,
    fun::Union{Nothing,Function}=nothing;
    dropdims=true,
    kwargs...,
) where {T}
    epreds = epred(TM, X; kwargs...) #don't pass fun
    ndraws = size(epreds, 2)
    if T == Normal
        σ = vec(get_parameters(TM, [:σ]; std=true, kwargs...))
        posterior_preds = (rand(T(), ndraws) .* σ)' .+ epreds
    elseif T == TDist
        σ = vec(get_parameters(TM, [:σ]; std=true, kwargs...))
        ν = vec(get_parameters(TM, [:ν]; std=true, kwargs...))
        posterior_preds = (rand.(T.(ν)) .* σ)' .+ epreds
    elseif T == Bernoulli
        posterior_preds = rand.(Bernoulli.(epreds))
    elseif T == Poisson
        posterior_preds = rand.(Poisson.(epreds))
    elseif T == NegativeBinomial
        ϕ⁻ = vec(get_parameters(TM, [:ϕ⁻]; std=true, kwargs...))
        posterior_preds = rand.(TuringGLM.NegativeBinomial2.(epreds, ϕ⁻'))
    end
    posterior_preds =
        isnothing(fun) ? posterior_preds : mapslices(fun, posterior_preds; dims=2)
    return dropdims ? drop_single_dims(posterior_preds) : posterior_preds
end
