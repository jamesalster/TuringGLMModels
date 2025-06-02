
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
    get_parameters(TM::TuringGLMModel, params::Vector{Symbol}; drop_warmup=200, n_draws=-1, collapse=true, kwargs...)

Extract specific parameters from fitted model as NamedArray.

# Arguments
- `params`: Vector of parameter symbols to extract
- `drop_warmup`: Number of warmup samples to drop from each chain
- `n_draws`: Number of draws to keep (-1 for all post-warmup)
- `collapse`: Whether to collapse chains into single dimension
"""
function get_parameters(TM::TuringGLMModel, params::Vector{Symbol}; kwargs...)
    isnothing(TM.samples) && throw(ArgumentError("Model has not been fitted."))
    new_names = parameter_names(TM, params)
    arr = NamedArray(TM.samples[params].value)
    setnames!(arr, new_names, 2)
    setdimnames!(arr, :draw, 1)
    params = process_draws(arr; kwargs...)
end

"""
    parameters(TM::TuringGLMModel, fun=nothing; drop_warmup=200, n_draws=-1, collapse=true, kwargs...)

Get all model parameters.

# Arguments
- `fun`: Optional function to apply across draws (e.g., mean, median)
- `drop_warmup`: Number of warmup samples to drop from each chain  
- `n_draws`: Number of draws to keep (-1 for all post-warmup)
- `collapse`: Whether to collapse chains into single dimension
"""
function parameters(TM::TuringGLMModel, fun::Union{Nothing, Function}=nothing; kwargs...)
    params = get_parameters(TM, TM.samples.name_map[:parameters]; kwargs...)
    params = isnothing(fun) ? params : mapslices(fun, params; dims = 1) 
    return drop_single_dims(params)
end

"""
    fixef(TM::TuringGLMModel, fun=nothing; drop_warmup=200, n_draws=-1, collapse=true, kwargs...)

Get fixed effect coefficients (β parameters).

# Arguments
- `fun`: Optional function to apply across draws (e.g., mean, median)
- `drop_warmup`: Number of warmup samples to drop from each chain
- `n_draws`: Number of draws to keep (-1 for all post-warmup)  
- `collapse`: Whether to collapse chains into single dimension
"""
function fixef(TM::TuringGLMModel, fun::Union{Nothing, Function}=nothing; kwargs...)
    fixef_names = [Symbol("β[$i]") for i in 1:size(TM.X, 2)]
    params = get_parameters(TM, fixef_names; kwargs...)
    params = isnothing(fun) ? params : mapslices(fun, params; dims = 1) 
    return drop_single_dims(params)
end

"""
    internals(TM::TuringGLMModel, fun=nothing; drop_warmup=200, n_draws=-1, collapse=true, kwargs...)

Get internal parameters (auxiliary parameters like σ, ν, etc).

# Arguments
- `fun`: Optional function to apply across draws (e.g., mean, median)
- `drop_warmup`: Number of warmup samples to drop from each chain
- `n_draws`: Number of draws to keep (-1 for all post-warmup)
- `collapse`: Whether to collapse chains into single dimension
"""
function internals(TM::TuringGLMModel, fun::Union{Nothing, Function}=nothing; kwargs...)
    internals_names = TM.samples.name_map[:internals]
    params = get_parameters(TM, internals_names; kwargs...)
    params = isnothing(fun) ? params : mapslices(fun, params; dims = 1) 
    return drop_single_dims(params)
end

"""
    coefs(TM::TuringGLMModel, fun=median)

Get coefficient point estimates using specified summary function.

# Arguments
- `fun`: Summary function to apply (default: median)
"""
function coefs(TM::TuringGLMModel, fun::Function=median)
    @info "Reducing with function: $(fun)"
    return fixef(TM, fun)
end

"""
    outcome(TM::TuringGLMModel)

Get the response variable as NamedArray.
"""
function outcome(TM::TuringGLMModel)
    return NamedArray(TM.y; names = (1:length(TM.y),), dimnames = (:row,))
end
