

# Internal function to create array of unstandardized parameters from the chain object
# Called on model fit
function _make_unstd_parameters!(TM::TuringGLMModel)
    unstd_params = copy(TM.samples.value)
    beta_vars = [Symbol("β[$i]") for i in 1:size(TM.X, 2)]
    # Do not transform internal values or shape values nu and phi
    if :σ ∈ TM.samples.name_map[:parameters]
        unstd_params[:,:σ, :] .*= TM.σ_y
    end
    # NB this relies on non-transformation of beta vars already happening
    correction_term = zeros(size(unstd_params[:, :α, :]))
    #dot product over dimension
    for chain in 1:size(TM.samples.value, 3)
        correction_term[:, chain] = unstd_params[:, beta_vars, chain] * (TM.μ_X ./ TM.σ_X)
    end
    unstd_params[:,:α, :] .= TM.μ_y .+ TM.σ_y .* (unstd_params[:,:α, :] .- correction_term)
    # Now transform beta vars
    unstd_params[:,beta_vars, :] .*= (TM.σ_y ./ TM.σ_X)'
    # Add to model object
    TM.unstd_params = Chains(
        unstd_params,
        TM.samples.logevidence,
        TM.samples.name_map,
        TM.samples.info
    )
end

# Internal funciton to unstandardize predictions
function unstandardize_predictions!(preds::AbstractArray, TM::TuringGLMModel)
    preds .*= TM.σ_y 
    preds .+= TM.μ_y
end

