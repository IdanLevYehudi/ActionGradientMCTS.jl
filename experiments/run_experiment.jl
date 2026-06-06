#!/usr/bin/env julia

using ArgParse
using ActionGradientMCTSExperiments

function parse_commandline(argv=ARGS; throw_on_error=false)
    settings = ArgParseSettings(
        description="Run ActionGradientMCTS paper experiments.",
        exc_handler=throw_on_error ? ArgParse.debug_handler : ArgParse.default_handler,
    )
    @add_arg_table! settings begin
        "--domain"
            help = "Experiment domain"
            arg_type = String
            range_tester = x -> x in keys(ActionGradientMCTSExperiments.RUNNER_FILES)
            required = true
        "--mode"
            help = "Experiment mode"
            arg_type = String
            range_tester = x -> x in ["sim", "ce-opt", "ablation"]
            required = true
        "--solver"
            help = "Solver names to run: dpw, ag-dpw, vpw, ag-vpw, pomcpow, vomcpow, or all"
            arg_type = String
            nargs = '+'
            default = ActionGradientMCTSExperiments.DEFAULT_SOLVERS
            range_tester = x -> x in SOLVER_CHOICES
        "--mdp"
            help = "Run the MDP variant where supported"
            action = :store_true
        "--simple"
            help = "Use simple MountainCar dynamics"
            action = :store_true
        "--D"
            help = "CollabLightDark state dimension"
            arg_type = Int
            default = 2
        "--K"
            help = "CollabLightDark agent count"
            arg_type = Int
            default = 1
        "--test-mode"
            help = "Use reduced experiment sizes"
            action = :store_true
    end
    return parse_args(argv, settings)
end

function runner_args(parsed)
    is_mdp = parsed["mdp"]
    solvers = normalize_solver_selection(parsed["solver"]; mdp=is_mdp)
    args = Dict{String,Any}(
        "sim" => parsed["mode"] == "sim",
        "ce-opt" => parsed["mode"] == "ce-opt",
        "ablation" => parsed["mode"] == "ablation",
        "mdp" => is_mdp,
        "simple" => parsed["simple"],
        "D" => parsed["D"],
        "K" => parsed["K"],
        "test-mode" => parsed["test-mode"],
        "solvers" => solvers,
    )
    return args
end

function main(argv=ARGS)
    parsed = parse_commandline(argv)
    include(joinpath(@__DIR__, "src", "runners", ActionGradientMCTSExperiments.runner_file(parsed["domain"])))
    return Main.main(runner_args(parsed))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
