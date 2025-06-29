
# build on MCMCChains.summarize
"""
    pretty(io::IO, TM::TuringGLMModel; funs=[median, std], quantiles=[0.025, 0.975], return_table=false, standardized=false, draws_idx=nothing, kwargs...)

Display formatted summary table of model parameters.

# Arguments
- `io`: Output stream 
- `TM`: Fitted TuringGLMModel
- `funs`: Summary functions to apply (default: [median, std])
- `standardized`: Return standardized results? Default is false
- `quantiles`: Quantiles to compute (default: [0.025, 0.975] for 95% CI)
- `return_table`: Whether to return the summary table as NamedTuple
- `draws_idx`: Subset of draws to use (default: all draws)
- `kwargs...`: Additional arguments passed to summarize
"""
function pretty(
    io::IO,
    TM::TuringGLMModel;
    funs=[median, std],
    quantiles=[0.025, 0.975],
    return_table=false,
    draws_idx=nothing,
    standardized=false,
    kwargs...,
)
    isnothing(TM.samples) && throw(ArgumentError("Turing Model has not yet been fit!()"))
    sample_obj = standardized ? TM.samples : TM.unstd_params

    funs_all = vcat(funs..., [(x -> quantile(x, q)) for q in quantiles]...)
    func_names_all = vcat(
        Symbol.(funs)..., [Symbol("q$(round(q*100; digits =1))") for q in quantiles]...
    )

    #draws idx
    draws_idx = something(draws_idx, 1:size(TM.samples, 1))
    #custom
    chain_df_funs = summarize(
        sample_obj[draws_idx, :, :],
        funs_all...;
        sections=:parameters,
        func_names=func_names_all,
    )
    #defaults
    chain_df_summary_stats = summarize(sample_obj[draws_idx, :, :]; sections=:parameters)
    # make as named tuple
    chain_info = (;
        chain_df_funs.nt[keys(chain_df_funs.nt)[2:end]]...,
        chain_df_summary_stats.nt[keys(chain_df_summary_stats.nt)[4:end]]...,
    )
    ncols = length(chain_info)

    #metrics
    drop_warmup = size(TM.samples, 1) < 400 ? 0 : 200
    metric_tabs = map(
        f -> default_metrics(TM, f; drop_warmup=drop_warmup, dropdims=false), funs_all
    )
    metric_tab = hcat(metric_tabs...)
    metric_tab = set(metric_tab, Dim{:draw} => Dim{:statistic})
    metric_tab = set(metric_tab, Dim{:statistic} => DimensionalData.Dimensions.Categorical)
    metric_tab = set(metric_tab, Dim{:statistic} => func_names_all)

    # show
    show(io, TM; warnings=false)
    println(io)
    pretty_table(
        io,
        chain_info;
        title="Fixed Effects",
        header=collect(keys(chain_info)),
        row_labels=parameter_names(TM),
        row_label_column_title="Parameter",
        highlighters=make_highlighters(ncols),
        formatters=(
            ft_printf("%5.2f", 1:(ncols - 5)),
            ft_printf("%5.2g", ncols-4),
            ft_printf("%5.0f", [ncols, ncols - 2, ncols - 3]),
            ft_printf("%5.3f", ncols - 1),
        ),
        default_options...,
    )
    pretty_table(
        io,
        parent(metric_tab);
        title="Prediction Metrics",
        header=Array(dims(metric_tab, 2)),
        row_labels=Array(dims(metric_tab, 1)),
        row_label_column_title="Metric",
        formatters=(ft_printf("%5.3f")),
        default_options...,
    )
    model_warnings(chain_info)
    if return_table
        return chain_info
    else
        return nothing
    end
end

# Catch-all method for non-IO calls
function pretty(TM::TuringGLMModel, args...; kwargs...)
    pretty(stdout, TM, args...; kwargs...)
end

function model_warnings(chain_info)
    #warnings
    if any(chain_info[:rhat] .> 1.05)
        @warn "Some rhat values are > 1.05, treat parameter estimates with caution!"
    elseif any(chain_info[:rhat] .> 1.01)
        @info "Note that some rhat values are > 1.01"
    end
    if any(chain_info[:ess_bulk] .< 100)
        @warn "Some parameters have bulk ess < 100, point estimates may be unreliable!"
    elseif any(chain_info[:ess_bulk] .< 250)
        @info "Note that some parameters have bulk ess < 250"
    end
    if any(chain_info[:ess_tail] .< 100)
        @warn "Some parameters have tail ess < 100, credible intervals may be unreliable!"
    elseif any(chain_info[:ess_tail] .< 250)
        @info "Note some parameters have tail ess < 250"
    end
    if :std ∈ keys(chain_info)
        if any((chain_info[:mcse] ./ chain_info[:std]) .> 0.05)
            @warn "MCSE is > 5% of standard error for some parameters!"
        elseif any((chain_info[:mcse] ./ chain_info[:std]) .> 0.01)
            @info "MCSE is > 1% of standard error for some parameters"
        end
    end
end

"""
    model_warnings(TM::TuringGLMModel)

Display info and warnings about model fit. Based on `MCMCChains.summarize`
"""
function model_warnings(TM::TuringGLMModel)
    isnothing(TM.samples) && return nothing
    chain_info = summarize(TM.samples; sections=:parameters)
    model_warnings(chain_info.nt)
end

# Highlighters
function make_highlighters(ncols)
    return (
        #R hat
        Highlighter(
            (data, i, j) -> (j == ncols-1 && data[j][i] > 1.05), crayon"bold magenta"
        ),
        Highlighter((data, i, j) -> (j == ncols-1 && data[j][i] > 1.02), crayon"magenta"),
        #ESS
        Highlighter(
            (data, i, j) -> (j ∈ [ncols-2, ncols-3] && data[j][i] < 100),
            crayon"bold magenta",
        ),
        Highlighter(
            (data, i, j) -> (j ∈ [ncols-2, ncols-3] && data[j][i] < 250), crayon"magenta"
        ),
    )
end

# Default table options
default_options = (;
    tf=tf_compact,
    header_crayon=crayon"bold",
    row_label_header_crayon=crayon"bold",
    crop=:horizontal,
    show_subheader=false,
)
