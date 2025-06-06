
"""
    conditional_dependency(TM::TuringGLMModel, variable::Symbol; type=:posterior, kwargs...)

Plot how predictions change when varying one variable while holding others at their means.  
    Kwargs are passed to Makie.Figure(). 
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

    preds = predict(TM, predgrid; type=type)

    # plot
    fig = Makie.Figure(kwargs...)
    ax = Makie.Axis(
        fig[1, 1];
        title="Conditional Dependency Plot",
        subtitle="Other variables held at their mean",
        ylabel="Outcome",
        xlabel=string(variable),
    )
    lineribbon!(ax, predgrid[:, id], preds')
    Makie.scatter!(ax, hcat(pp[:, id], outcome(TM)))
    fig
end

"""
    pp_check_hist(TM::TuringGLMModel; bins=20, type=:posterior, kwargs...)

Compare predicted vs observed values using histograms. Kwargs are passed to Makie.Figure().
"""
function pp_check_hist(TM::TuringGLMModel; bins=20, type=:posterior, kwargs...)
    preds = predict(TM, median; type=type)
    fig = Makie.Figure(kwargs...)
    ax = Makie.Axis(fig[1, 1]; title="Posterior Predictive Check", xlabel="Outcome")
    Makie.hist!(ax, preds; label="Predictions", bins=bins)
    Makie.hist!(ax, outcome(TM); label="Data", bins=bins)
    Makie.axislegend(ax; position=:rt)
    fig
end

"""
    pp_check_dens(TM::TuringGLMModel; bandwidth = Makie.automatic, type=:posterior, kwargs...)

Compare predicted vs observed values using density curves. Kwargs are passed to Makie.Figure().
"""
function pp_check_dens(
    TM::TuringGLMModel; bandwidth=Makie.automatic, type=:posterior, kwargs...
)
    preds = predict(TM, median; type=type)
    fig = Makie.Figure(kwargs...)
    ax = Makie.Axis(fig[1, 1]; title="Posterior Predictive Check", xlabel="Outcome")
    Makie.density!(ax, preds; bandwidth=bandwidth, label="Predictions")
    Makie.density!(ax, outcome(TM); bandwidth=bandwidth, label="Data")
    Makie.axislegend(ax; position=:rt)
    fig
end

"""
    pp_check_dens_overlay(TM::TuringGLMModel; n_draws=100, type=:posterior, kwargs...)

Overlay multiple prediction density curves against observed data density. Kwargs are passed to Makie.Figure().
"""
function pp_check_dens_overlay(TM::TuringGLMModel; n_draws=100, type=:posterior, kwargs...)
    preds = predict(TM; n_draws=n_draws, type=type)
    fig = Makie.Figure(kwargs...)
    ax = Makie.Axis(fig[1, 1]; title="Posterior Predictive Check", xlabel="Outcome")
    for i in 1:n_draws
        Makie.density!(
            ax,
            preds[:, i];
            color="#FFFFFF00",
            alpha=0.2,
            strokewidth=1,
            strokecolor=:dodgerblue,
            strokearound=true,
        )
    end
    Makie.density!(
        ax,
        outcome(TM);
        color="#FFFFFF00",
        strokewidth=2,
        strokecolor=:dodgerblue4,
        strokearound=true,
    )
    fig
end
