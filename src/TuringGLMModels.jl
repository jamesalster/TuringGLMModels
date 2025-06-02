
module TuringGLMModels

using Reexport
@reexport using TuringGLM
@reexport using NamedArrays
@reexport using LogExpFunctions: logit, logistic
using PrettyTables
using GLM: FormulaTerm
using DynamicPPL: Model
using Suppressor: @suppress
using ParetoSmooth

include("turingglmmodel.jl")
include("utils.jl")
include("parametermethods.jl")
include("predict.jl")
include("extend/turing_model.jl")
include("pretty.jl")
include("comparison.jl")

export
    TuringGLMModel,
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
    predict,
    linpred,
    epred,
    posterior_pred,
    psis_loo,
    loo_compare

end