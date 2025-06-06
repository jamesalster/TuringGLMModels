
"""
    conditional_dependency(TM::TuringGLMModel, variable::Symbol; type=:posterior, kwargs...)

Plot how predictions change when varying one variable while holding others at their means.
"""
function conditional_dependency(
    TM::TuringGLMModel, variable::Symbol; type=:posterior, kwargs...
)
    N = 200
    pp = predictors(TM)
    id = findfirst(==(variable), TM.X_names)
    means = mean(pp; dims=1)
    predict_range = range.(extrema(pp[var = At(variable)])..., N)

    # make prediction_grid
    predgrid = zeros(N, length(means))
    for i in eachindex(means)
        if i == id
            predgrid[:, i] = collect(predict_range)
        else
            predgrid[:, i] = fill(means[i], N)
        end
    end

    preds = predict(mod, predgrid; type=type)

    # plot
    fig = Figure(kwargs...)
    ax = Axis(fig[1, 1])
    lineribbon!(predgrid[:, id], preds')
    scatter!(hcat(pp[:, 2], outcome(mod)))
    fig
end

"""
    pp_check_hist(TM::TuringGLMModel; bins=20, type=:posterior, kwargs...)

Compare predicted vs observed values using histograms.
"""
function pp_check_hist(TM::TuringGLMModel; bins=20, type=:posterior, kwargs...)
    preds = predict(mod, median; type=type)
    fig = Figure()
    ax = Axis(fig[1, 1]; title="Posterior Predictive Check")
    hist!(preds; label="Predictions", bins=bins)
    hist!(outcome(mod); label="Outcome", bins=bins)
    axislegend(ax; position=:rt)
    fig
end

"""
    pp_check_dens(TM::TuringGLMModel; bandwidth = Makie.automatic, type=:posterior, kwargs...)

Compare predicted vs observed values using density curves.
"""
function pp_check_dens(
    TM::TuringGLMModel; bandwidth=Makie.automatic, type=:posterior, kwargs...
)
    preds = predict(mod, median; type=type)
    fig = Figure()
    ax = Axis(fig[1, 1]; title="Posterior Predictive Check")
    density!(preds; bandwidth=bandwidth, label="Predictions")
    density!(outcome(mod); bandwidth=bandwidth, label="Outcome")
    axislegend(ax; position=:rt)
    fig
end

"""
    pp_check_dens_overlay(TM::TuringGLMModel; n_draws=100, type=:posterior, kwargs...)

Overlay multiple prediction density curves against observed data density.
"""
function pp_check_dens_overlay(TM::TuringGLMModel; n_draws=100, type=:posterior, kwargs...)
    preds = predict(mod; n_draws=100, type=type)
    fig = Figure()
    ax = Axis(fig[1, 1]; title="Posterior Predictive Check")
    for i in 1:n_draws
        density!(
            preds[:, i];
            color="#FFFFFF00",
            alpha=0.2,
            strokewidth=1,
            strokecolor=:dodgerblue,
            strokearound=true,
        )
    end
    density!(
        outcome(mod);
        color="#FFFFFF00",
        strokewidth=2,
        strokecolor=:dodgerblue4,
        strokearound=true,
    )
    fig
end
