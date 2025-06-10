
module TuringGLMModels

using Reexport
using Requires: @require
@reexport using TuringGLM
@reexport using DimensionalData
@reexport using LogExpFunctions: logit, logistic
using PrettyTables
using StatisticalMeasures
using ParetoSmooth

using GLM: FormulaTerm
using DynamicPPL: Model
using Suppressor: @suppress
using StatsBase: mean, std
using DataFrames: DataFrame
using Colors: colormap
using CategoricalArrays: categorical
using CategoricalDistributions: UnivariateFinite

include("turingglmmodel.jl")
include("utils.jl")
include("unstandardize.jl")
include("parametermethods.jl")
include("predict.jl")
include("extend/turing_model.jl")
include("pretty.jl")
include("metrics.jl")
include("comparison.jl")

export TuringGLMModel,
    turing_glm,
    fit!,
    pretty,
    model_warnings,
    parameter_names,
    get_parameters,
    parameters,
    coefs,
    fixef,
    internals,
    outcome,
    predictors,
    outcome_as_distribution,
    predict,
    psis_loo,
    loo_compare,
    lineribbon,
    calculate_metrics,
    default_metrics

function __init__()
    #Makie required for band
    @require Makie="ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" begin
        include("plots/lineribbon.jl")
        export lineribbon
        include("plots/plots.jl")
        export conditional_dependency, pp_check_dens, pp_check_dens_overlay, pp_check_hist
    end
end
end
