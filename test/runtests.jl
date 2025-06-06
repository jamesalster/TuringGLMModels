
using TuringGLMModels
using Test
using RDatasets
using MCMCChains
using ParetoSmooth
using StatsModels
using StatsBase: mean, std
using Suppressor: @suppress
using Random

@info "Setting up tests"
mtcars = dataset("datasets", "mtcars")

@warn "No tests yet implemented for actual fit/prediction correctness."


@testset "Model Creation" begin

    mod1 = @test_nowarn turing_glm(
        @formula(MPG ~ Cyl + Disp), 
        mtcars, 
        Normal
    );

    predmat = Matrix(mtcars[:,[:Cyl, :Disp]])
    y = mtcars.MPG

    @test mod1.formula isa StatsModels.FormulaTerm
    @test mod1.model isa DynamicPPL.Model
    @test mod1.prior isa CustomPrior
    @test mod1.link == identity
    @test mod1.y == y
    @test mod1.X == predmat
    @test isnothing(mod1.Z)
    @test mod1.X_names == (:Cyl, :Disp)
    @test mod1.Z_names == ()
    @test !mod1.standardized
    @test isnothing(mod1.samples)

    # CUstrom prior
    prior = CustomPrior(Normal(0,2), Normal(20, 5), Exponential(1));
    mod1b = @test_nowarn turing_glm(@formula(MPG ~ Cyl + Disp), mtcars, Normal; priors=prior);
    @test mod1b.prior == prior

    # Test X y method
    mod2 = @test_nowarn turing_glm(
        y, predmat, Normal;
        names=[:Cyl, :Disp]
    );
    @test mod2.formula isa StatsModels.FormulaTerm
    @test mod2.y == mod1.y
    @test mod2.X == mod1.X
    @test mod2.X_names == mod1.X_names

    mod2b = @test_nowarn turing_glm(y, predmat, Normal)
    @test mod2b.X_names == (:X1, :X2)
    @test mod2b.X == mod1.X

    # Model creation
    mod3 = @test_throws ArgumentError turing_glm( @formula(MPG ~ Cyl + (1|Disp)), mtcars, Normal);
    
    # Link
    mod3 = turing_glm(@formula(MPG ~ Cyl + Disp), mtcars, TDist);
    @test mod3.link == identity
    mod4 = turing_glm(@formula(MPG ~ Cyl + Disp), mtcars, Bernoulli);
    @test mod4.link == logit
    mod5 = turing_glm(@formula(MPG ~ Cyl + Disp), mtcars, Poisson);
    @test mod5.link == log
    mod6 = turing_glm(@formula(MPG ~ Cyl + Disp), mtcars, NegativeBinomial);
    @test mod6.link == log
    # Test error (from TuringGLM) on wrong family
    mod7 = @test_throws ArgumentError turing_glm(@formula(MPG ~ Cyl + Disp), mtcars, Categorical)

    # Test standardized warning
    mod2 = @test_warn "standardized" turing_glm( @formula(MPG ~ Cyl + Disp), mtcars, Normal; standardize=true )

    @test mod2.standardized
    @test isapprox(mean(mod2.X; dims = 1), zeros(1, size(mod2.X, 2)); atol=1e-12)
    @test isapprox(std(mod2.X; dims = 1), ones(1, size(mod2.X, 2)); atol=1e-12)
end

@testset "Model Fit" begin
    mod1 = turing_glm( @formula(MPG ~ Cyl + Disp), mtcars, Normal);n

    # Fit
    mod1 = @test_nowarn fit!(mod1);
    @test mod1.samples isa MCMCChains.Chains

    # Default size
    @test size(mod1.samples) == (2000, 16, 4)

    # Kwargs to fit
    @test_nowarn fit!(mod1; parallel=MCMCSerial(), N = 4000, nchains=1);
    @test size(mod1.samples) == (4000, 16, 1)

    # Fit warnings
    mod1 = fit!(mod1, N=10);
    @test_warn "rhat" show(mod1);
    @test_warn "ess" show(mod1);
    @test_warn "MCSE" show(mod1);
end

# Init global model for following sections
@info "Fitting models for tests"

prior = CustomPrior(Normal(0,2), Normal(20, 5), Exponential(1));
mod = turing_glm(@formula(MPG ~ Cyl + Disp), mtcars, TDist; priors=prior);
mod_empty = deepcopy(mod);
mod = @suppress fit!(mod);
mod_std = @suppress turing_glm(@formula(MPG ~ Cyl + Disp), mtcars, Normal; standardize=true);
mod_std = @suppress fit!(mod_std, N=50, nchains=1);
mod_count = turing_glm(@formula(HP ~ Cyl + Disp), mtcars, Poisson);
fit!(mod_count);

@testset "Parameter Methods" begin

    default_samples = 2000
    default_chains = 4
    default_dropwarmup = 200
    expected_out = (default_samples - default_dropwarmup) * default_chains

    param_names = [:α, :Cyl, :Disp, :σ, :ν]
    @test parameter_names(mod) == param_names

    # Test the main get_parameters method
    pars = [:α, :σ, :ν]
    ps = get_parameters(mod, pars)
    @test ps isa DimArray
    @test size(ps) == (expected_out, length(pars))
    @test Array(dims(ps, 2)) == string.(pars)
    expected_idx = vec([(i, j) for i in default_dropwarmup+1:default_samples, j in 1:default_chains])
    @test Array(dims(ps, 1)) == expected_idx

    # Test kwargs
    @test size(get_parameters(mod, pars; drop_warmup = 0)) == (default_samples * default_chains, length(pars))
    @test size(get_parameters(mod, pars; drop_warmup = 800)) == ((default_samples - 800) * default_chains, length(pars))
    @test size(get_parameters(mod, pars; n_draws = 50)) == (50 * default_chains, length(pars))
    @test size(get_parameters(mod, pars; n_draws = 50)) == (50 * default_chains, length(pars))
    @test size(get_parameters(mod, pars; n_draws = 50)) == (50 * default_chains, length(pars))
    @test size(get_parameters(mod, pars; collapse=false)) == (default_samples - default_dropwarmup, length(pars), default_chains)
    @test_throws ErrorException get_parameters(mod, pars; drop_warmup = 2000, n_draws=5000) 

    # Test derivative methods with funciton and kwargs
    pars = parameters(mod)
    @test pars == get_parameters(mod, [:α, Symbol("β[1]"),Symbol("β[2]"), :σ, :ν])
    @test isapprox(parameters(mod, mean), transpose(mean(parameters(mod), dims=1)))
    @test ndims(parameters(mod, median; dropdims=false)) == 2
    @test ndims(parameters(mod, median; collapse=false, dropdims=false)) == 3

    # On single chain version with too few draws
    @test_warn "drop_warmup" get_parameters(mod_std, [:α])
    @test ndims(parameters(mod_std, median; drop_warmup=0, collapse=false, dropdims=false)) == 3
    @test ndims(parameters(mod_std, median; drop_warmup=0, collapse=false, dropdims=true)) == 1

    # Test other methods more simply
    @test fixef(mod) == get_parameters(mod, [Symbol("β[1]"),Symbol("β[2]")])
    @test isapprox(fixef(mod, mean), transpose(mean(fixef(mod), dims=1)))
    @test ndims(fixef(mod, median; dropdims=false)) == 2

    @test coefs(mod) == fixef(mod, median)

    ints = internals(mod)
    @test Array(dims(ints, 2)) == [
        "lp",
        "n_steps",
        "is_accept",
        "acceptance_rate",
        "log_density",
        "hamiltonian_energy",
        "hamiltonian_energy_error",
        "max_hamiltonian_energy_error",
        "tree_depth",
        "numerical_error",
        "step_size",
        "nom_step_size",
    ]
    @test isapprox(internals(mod, mean), transpose(mean(ints, dims=1)))
    @test ndims(internals(mod, median; dropdims=false)) == 2

    out = outcome(mod) 
    @test out isa DimArray
    @test out == mod.y
end

@testset "Prediction" begin
    # Basic prediction, with a count model
    pred_data = Matrix(mtcars[5:9,[:Cyl, :Disp]])

    #basic linpred
    alpha = vec(Array(get_parameters(mod_count, [:α])))
    beta = Matrix(get_parameters(mod_count, [Symbol("β[1]"),Symbol("β[2]")]))
    pred_data = Matrix(mtcars[5:9,[:Cyl, :Disp]])
    pred_manual = alpha .+ beta * pred_data' 
    @test linpred(mod_count, pred_data) == transpose(pred_manual)

    # Methods pass
    lp = @test_nowarn linpred(mod_count, pred_data)
    ep = @test_nowarn epred(mod_count, pred_data)
    pp = @test_nowarn posterior_pred(mod_count, pred_data)

    # Relations: link
    @test isapprox(lp, log.(ep))
    @test var(pp) > var(ep) #higher variance
    # Relations: without link
    @test linpred(mod, pred_data) == epred(mod, pred_data)
    @test var(posterior_pred(mod, pred_data)) > var(epred(mod, pred_data))

    # calling self
    @test predict(mod_count, type=:linpred) == linpred(mod_count, mod_count.X)
    @test predict(mod_count, mod.X, type=:linpred) == linpred(mod_count, mod_count.X)
    @test predict(mod_count, type=:epred) == epred(mod_count, mod_count.X)
    @test let
        Random.seed!(349)
        pp1 = predict(mod_count, type=:posterior) 
        Random.seed!(349)
        pp2 = posterior_pred(mod_count, mod_count.X)
        pp1 == pp2
    end
end

@testset "Display Methods" begin
    show_output = sprint(show, mod)
    @test contains(show_output, "TuringGLM Model")
    @test contains(show_output, "TDist (link: identity)")
    @test contains(show_output, "MPG ~ Cyl + Disp")
    @test contains(show_output, "Prior")
    @test contains(show_output, "Predictors")
    @test contains(show_output, "Intercept")
    @test contains(show_output, "Intercept")
    @test contains(show_output, "Normal(μ=0.0, σ=2.0)")
    @test contains(show_output, "Normal(μ=20.0, σ=5.0)")
    @test contains(show_output, "Exponential(θ=1.0)")
    @test contains(show_output, "Auxiliary")
    @test contains(show_output, "TDist")
    @test contains(show_output, "32") #Observations
    @test contains(show_output, "8000 samples") 
    @test contains(show_output, "4 chains") 
    # Standardized
    @test contains(sprint(show, mod_std), "standardized")
    # Empty
    @test contains(sprint(show, mod_empty), "empty")

    # Pretty version
    pretty_output = sprint(pretty, mod)
    @test contains(pretty_output, show_output)
    @test contains(pretty_output, "Fixed Effects")
    @test all(contains.(Ref(pretty_output), ["median", "std", "q2.5", "q97.5", "mcse", "ess_bulk", "ess_tail"]))
    @test all(contains.(Ref(pretty_output), ["Cyl", "Disp"]))
    coef = string.(round.(parent(parameters(mod, median; drop_warmup = 0)); digits = 2))
    @test all(contains.(Ref(pretty_output), coef))
    # that pretty and show contain warning
    @test_warn "rhat" show(mod_std);
    @test_warn "rhat" pretty(mod_std);
end

@testset "Model Comparison" begin
    @test psis_loo(mod) isa ParetoSmooth.PsisLoo
    comp1 = loo_compare(mod, mod_std)
    @test comp1 isa ParetoSmooth.ModelComparison
    comp2 = loo_compare([mod, mod_std])
    @test sprint(show, comp1) == sprint(show, comp2)
    comp1b = loo_compare(mod, mod_std; model_names = ("Mod1", "Mod2"))
    @test contains(sprint(show, comp1b), "Mod1")
end
