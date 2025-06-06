
#API for ParetoSmooth

"""
    psis_loo(TM::TuringGLMModel)

Calculate leave-one-out cross-validation using Pareto smoothed importance sampling.

Returns PSIS-LOO object with predictive accuracy measures. Lower ELPD values indicate better predictive performance.
"""
function ParetoSmooth.psis_loo(TM::TuringGLMModel)
    ll = loglikelihood(TM.model, TM.samples)
    ll_rshp = reshape(ll, 1, size(ll)...)
    return psis_loo(ll_rshp; source="mcmc")
end

"""
    loo_compare(models::AbstractVector{<:TuringGLMModel}; kwargs...)

Compare multiple models using leave-one-out cross-validation.
    Passing `model_names` as a tuple will name the outputs.

# Arguments
- `models`: Vector of fitted TuringGLMModel objects
- `kwargs...`: Additional arguments passed to ParetoSmooth.loo_compare
"""
function ParetoSmooth.loo_compare(models::AbstractVector{<:TuringGLMModel}; kwargs...)
    @nospecialize models
    psis_objects = psis_loo.(models)
    return loo_compare(psis_objects; kwargs...)
end

"""
    loo_compare(models::TuringGLMModel...; kwargs...)

Compare multiple models using leave-one-out cross-validation.
    Passing `model_names` as a tuple will name the outputs.

# Arguments
- `models...`: Multiple TuringGLMModel objects passed as separate arguments
- `kwargs...`: Additional arguments passed to ParetoSmooth.loo_compare
"""
function ParetoSmooth.loo_compare(models::TuringGLMModel...; kwargs...)
    @nospecialize models
    @views models = [models[i] for i in 1:length(models)]
    cv_results = psis_loo.(models)
    return loo_compare(cv_results; kwargs...)
end
