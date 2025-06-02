
# build on MCMCChains.summarize
"""
    pretty(io::IO, TM::TuringGLMModel; funs=[median, std], quantiles=[0.025, 0.975], return_table=false, draws_idx=nothing, kwargs...)

Display formatted summary table of model parameters.

# Arguments
- `io`: Output stream 
- `TM`: Fitted TuringGLMModel
- `funs`: Summary functions to apply (default: [median, std])
- `quantiles`: Quantiles to compute (default: [0.025, 0.975] for 95% CI)
- `return_table`: Whether to return the summary table as NamedTuple
- `draws_idx`: Subset of draws to use (default: all draws)
- `kwargs...`: Additional arguments passed to summarize
"""
function pretty(io::IO, TM::TuringGLMModel; funs = [median, std], quantiles = [0.025, 0.975], return_table=false, draws_idx=nothing, kwargs...)
    isnothing(TM.samples) && throw(ArgumentError("Turing Model has not yet been fit!()"))
    funs_all = vcat(funs..., [(x -> quantile(x, q)) for q in quantiles]...)
    func_names_all = vcat(Symbol.(funs)..., [Symbol("q$(round(q*100; digits =1))") for q in quantiles]...)
    #draws idx
    draws_idx = something(draws_idx, 1:size(TM.samples, 1))
    #custom
    chain_df_funs = summarize(TM.samples[draws_idx, :, :], funs_all...; sections = :parameters, func_names = func_names_all)
    #defaults
    chain_df_summary_stats = summarize(TM.samples[draws_idx, :, :]; sections = :parameters)
    # make as named tuple
    chain_info = (; 
        chain_df_funs.nt[keys(chain_df_funs.nt)[2:end]]...,
        chain_df_summary_stats.nt[keys(chain_df_summary_stats.nt)[4:end]]...,)
    ncols = length(chain_info)

    # show
    show(io, TM; warnings=false)
    println(io)
    pretty_table(
        io,
        chain_info;
        title = "Fixed Effects",
        header=collect(keys(chain_info)),
        row_labels=parameter_names(TM),
        row_label_column_title="Parameter",
        highlighters = make_highlighters(ncols),
        formatters=(
            ft_printf("%5.2f", 1:ncols-5), 
            ft_printf("%5.2g", ncols-4), 
            ft_printf("%5.0f", [ncols, ncols - 2, ncols -3]),
            ft_printf("%5.3f", ncols - 1)
        ),
        tf = tf_compact,
        header_crayon=crayon"bold",
        row_label_header_crayon=crayon"bold",
        crop=:horizontal,
        show_subheader=false
    )
    model_warnings(chain_info)
    if return_table return chain_info else return nothing end
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
    chain_info = summarize(TM.samples; sections = :parameters).nt
    model_warnings(chain_info)
end

# Highlighters
function make_highlighters(ncols)
    return (
        #R hat
        Highlighter((data, i, j) -> (j == ncols-1 && data[j][i] > 1.05), crayon"bold magenta"),
        Highlighter((data, i, j) -> (j == ncols-1 && data[j][i] > 1.02), crayon"magenta"),
        #ESS
        Highlighter((data, i, j) -> (j ∈ [ncols-2, ncols-3] && data[j][i] < 100), crayon"bold magenta"),
        Highlighter((data, i, j) -> (j ∈ [ncols-2, ncols-3] && data[j][i] < 250), crayon"magenta"),
    )
end

