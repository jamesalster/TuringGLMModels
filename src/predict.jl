
"""
    predict(TM::TuringGLMModel, X::AbstractArray, fun=nothing; type=:posterior, kwargs...)
    predict(TM::TuringGLMModel, fun=nothing; kwargs...)

Generate predictions for new data or fitted data.

# Arguments
- `X`: Design matrix for predictions (optional, uses fitted data if omitted)
- `fun`: Optional function to apply across draws
- `type`: Type of prediction (:posterior, :epred, :linpred)
"""
function predict(TM::TuringGLMModel, X::AbstractArray, fun::Union{Nothing, Function}=nothing; type::Symbol = :posterior, kwargs...)
    #TODO handle Z
    if type === :posterior
        return posterior_pred(TM, X, fun; kwargs...)
    elseif type === :epred
        return epred(TM, X, fun; kwargs...)
    elseif type === :linpred
        return linpred(TM, X, fun; kwargs...)
    else
        throw(ArgumentError("type must be one of :posterior, :epred or :linpred"))
    end
end

function predict(TM::TuringGLMModel, fun::Union{Nothing, Function}=nothing; kwargs...)
    #TODO handle Z
    return predict(TM, TM.X, fun; kwargs...)
end

"""
    linpred(TM::TuringGLMModel, X::AbstractArray, fun=nothing; drop_warmup=200, n_draws=-1, collapse=true, kwargs...)

Generate linear predictor values (α + Xβ).

# Arguments
- `X`: Design matrix
- `fun`: Optional function to apply across draws
- `drop_warmup`: Number of warmup samples to drop from each chain
- `n_draws`: Number of draws to keep (-1 for all post-warmup)
- `collapse`: Whether to collapse chains into single dimension
"""
function linpred(TM::TuringGLMModel, X::AbstractArray, fun::Union{Nothing, Function}=nothing; kwargs...)
    #standardize? - check here since this is the 'base' method for all predictions
    TM.standardized && (X !== TM.X) && @warn "Standardization not applied to new data when predicting."

    α = get_parameters(TM, [:α]; kwargs...) # vec required for NamedArray problems below
    β = fixef(TM; kwargs...)
    # handle 3d
    μ = zeros(eltype(α), (Dim{:row}(size(X, 1)), Dim{:draw}(size(α, 1)), Dim{:chain}(size(α, 3))))
    for i in 1:size(μ, 3)
        μ[:, :, i] = vec(α[:, :, i])' .+ X * β[:, :, i]'
    end
    μ = isnothing(fun) ? μ : mapslices(fun, μ; dims = 2) 
    return drop_single_dims(μ)
end

"""
    epred(TM::TuringGLMModel, X::AbstractArray, fun=nothing; drop_warmup=200, n_draws=-1, collapse=true, kwargs...)

Generate expected predictions (inverse link of linear predictor).

# Arguments  
- `X`: Design matrix
- `fun`: Optional function to apply across draws
- `drop_warmup`: Number of warmup samples to drop from each chain
- `n_draws`: Number of draws to keep (-1 for all post-warmup)
- `collapse`: Whether to collapse chains into single dimension
"""
function epred(TM::TuringGLMModel{T}, X::AbstractArray, fun::Union{Nothing, Function}=nothing; kwargs...) where T
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
    epreds = isnothing(fun) ? epreds : mapslices(fun, epreds; dims = 2) 
    return drop_single_dims(epreds)
end

"""
    posterior_pred(TM::TuringGLMModel, X::AbstractArray, fun=nothing; drop_warmup=200, n_draws=-1, collapse=true, kwargs...)

Generate posterior predictive samples (includes observation noise).

# Arguments
- `X`: Design matrix  
- `fun`: Optional function to apply across draws
- `drop_warmup`: Number of warmup samples to drop from each chain
- `n_draws`: Number of draws to keep (-1 for all post-warmup)
- `collapse`: Whether to collapse chains into single dimension
"""
function posterior_pred(TM::TuringGLMModel{T}, X::AbstractArray, fun::Union{Nothing, Function}=nothing; kwargs...) where T
    epreds = epred(TM, X; kwargs...) #don't pass fun
    ndraws = size(epreds, 2)
    if T == Normal
        σ = vec(get_parameters(TM, [:σ]; kwargs...))
        posterior_preds = (rand(T(), ndraws) .* σ)' .+ epreds
    elseif T == TDist
        σ = vec(get_parameters(TM, [:σ]; kwargs...))
        ν = vec(get_parameters(TM, [:ν]; kwargs...))
        posterior_preds = (rand.(T.(ν)) .* σ)' .+ epreds
    elseif T == Bernoulli
        posterior_preds = rand.(Bernoulli.(epreds))
    elseif T == Poisson
        posterior_preds = rand.(Poisson.(epreds))
    elseif T == NegativeBinomial
        ϕ⁻ = vec(get_parameters(TM, [:ϕ⁻]; kwargs...))
        posterior_preds = rand.(TuringGLM.NegativeBinomial2.(epreds, ϕ⁻'))
    end
    posterior_preds = isnothing(fun) ? posterior_preds : mapslices(fun, posterior_preds; dims = 2) 
    return drop_single_dims(posterior_preds)
end