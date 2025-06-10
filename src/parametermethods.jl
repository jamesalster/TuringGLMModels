
# Internal function to access parameters from samples object as DimArray
function _get_parameters(TM::TuringGLMModel, params::Vector{Symbol})::DimArray
    isnothing(TM.samples) && throw(ArgumentError("Model has not been fitted."))
    arr = DimArray(TM.samples[params].value, (Dim{:draw}, Dim{:param}(params), Dim{:chain}))
    return arr
end

# Internal function to access parameters from unstandardized samples object as DimArray
function _get_unstd_parameters(TM::TuringGLMModel, params::Vector{Symbol})::DimArray
    isnothing(TM.samples) && throw(ArgumentError("Model has not been fitted."))
    arr = DimArray(
        TM.unstd_params[params].value, (Dim{:draw}, Dim{:param}(params), Dim{:chain})
    )
    return arr
end

## Parameter methods
"""
    parameter_names(TM::TuringGLMModel, params=TM.samples.name_map[:parameters])

Get parameter names with friendly labels replacing generic β indices.
"""
function parameter_names(TM::TuringGLMModel, params=TM.samples.name_map[:parameters])
    rename_dict = Dict(Symbol("β[$i]") => nm for (i, nm) in enumerate(TM.X_names))
    return [get(rename_dict, p, p) for p in params]
end

"""
    get_parameters(TM::TuringGLMModel, params::Vector{Symbol}; std=false, drop_warmup=200, n_draws=-1, collapse=true, kwargs...)

Extract specific parameters from fitted model as DimArray.

# Arguments
- `params`: Vector of parameter symbols to extract
- `std`: Show standardized coefficients, or scaled back to the original data? Default=false.
- `drop_warmup`: Number of warmup samples to drop from each chain
- `n_draws`: Number of draws to keep (-1 for all post-warmup)
- `collapse`: Whether to collapse chains into single dimension
"""
function get_parameters(
    TM::TuringGLMModel, params::Vector{Symbol}; std=false, kwargs...
)::DimArray
    isnothing(TM.samples) && throw(ArgumentError("Model has not been fitted."))
    # access parameters
    arr = std ? _get_parameters(TM, params) : _get_unstd_parameters(TM, params)
    # rename
    new_names = string.(parameter_names(TM, params)) # string to allow regex lookup
    arr = set(arr, Dim{:param} => new_names)
    # Filter draws
    arr = process_draws(arr; kwargs...)
    size(arr, 1) == 0 &&
        @warn "No samples returned, check kwargs and perhaps try adjusting `drop_warmup`?"
    return arr
end

"""
    parameters(TM::TuringGLMModel, fun=nothing; drop_warmup=200, n_draws=-1, collapse=true, dropdims=true, kwargs...)

Get all model parameters.

# Arguments
- `fun`: Optional function to apply across draws (e.g., mean, median)
- `drop_warmup`: Number of warmup samples to drop from each chain  
- `n_draws`: Number of draws to keep (-1 for all post-warmup)
- `collapse`: Whether to collapse chains into single dimension
- `dropdims`: Whether to drop singleton dimensions (default: true)
- `std`: Show standardized coefficients, or scaled back to the original data? Default=false.
"""
function parameters(
    TM::TuringGLMModel, fun::Union{Nothing,Function}=nothing; dropdims=true, kwargs...
)
    params = get_parameters(TM, TM.samples.name_map[:parameters]; kwargs...)
    params = isnothing(fun) ? params : mapslices(fun, params; dims=1)
    return dropdims ? drop_single_dims(params) : params
end

"""
    fixef(TM::TuringGLMModel, fun=nothing; drop_warmup=200, n_draws=-1, collapse=true, dropdims=true, kwargs...)

Get fixed effect coefficients (β parameters).

# Arguments
- `fun`: Optional function to apply across draws (e.g., mean, median)
- `drop_warmup`: Number of warmup samples to drop from each chain
- `n_draws`: Number of draws to keep (-1 for all post-warmup)  
- `collapse`: Whether to collapse chains into single dimension
- `dropdims`: Whether to drop singleton dimensions (default: true)
- `std`: Show standardized coefficients, or scaled back to the original data? Default=false.
"""
function fixef(
    TM::TuringGLMModel, fun::Union{Nothing,Function}=nothing; dropdims=true, kwargs...
)
    fixef_names = [:α, [Symbol("β[$i]") for i in 1:size(TM.X, 2)]...]
    params = get_parameters(TM, fixef_names; kwargs...)
    params = isnothing(fun) ? params : mapslices(fun, params; dims=1)
    return dropdims ? drop_single_dims(params) : params
end

"""
    internals(TM::TuringGLMModel, fun=nothing; drop_warmup=200, n_draws=-1, collapse=true, dropdims=true, kwargs...)

Get internal parameters (auxiliary parameters like σ, ν, etc).

# Arguments
- `fun`: Optional function to apply across draws (e.g., mean, median)
- `drop_warmup`: Number of warmup samples to drop from each chain
- `n_draws`: Number of draws to keep (-1 for all post-warmup)
- `collapse`: Whether to collapse chains into single dimension
- `dropdims`: Whether to drop singleton dimensions (default: true)
- `std`: Show standardized coefficients, or scaled back to the original data? Default=false.
"""
function internals(
    TM::TuringGLMModel, fun::Union{Nothing,Function}=nothing; dropdims=true, kwargs...
)
    internals_names = TM.samples.name_map[:internals]
    params = get_parameters(TM, internals_names; kwargs...)
    params = isnothing(fun) ? params : mapslices(fun, params; dims=1)
    return dropdims ? drop_single_dims(params) : params
end

"""
    coefs(TM::TuringGLMModel, fun=median)

Get coefficient point estimates using specified summary function.

# Arguments
- `fun`: Summary function to apply (default: median)
- `std`: Show standardized coefficients, or scaled back to the original data? Default=false.
"""
function coefs(TM::TuringGLMModel, fun::Function=median; kwargs...)
    @info "Reducing with function: $(fun)"
    return fixef(TM, fun; kwargs...)
end
