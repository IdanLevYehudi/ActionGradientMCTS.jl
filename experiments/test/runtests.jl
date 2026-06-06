using Test
using ActionGradientMCTSExperiments

include(joinpath(dirname(@__DIR__), "run_experiment.jl"))

parsed_commandline(args) = parse_commandline(args; throw_on_error=true)
parsed_runner_args(args) = runner_args(parsed_commandline(args))

@testset "ActionGradientMCTSExperiments" begin
    expected_runner_files = Dict(
        "mountain-car" => "MountainCarRunner.jl",
        "lunar-lander" => "LunarLanderRunner.jl",
        "collab-light-dark" => "CollabLightDarkRunner.jl",
    )

    @test ActionGradientMCTSExperiments.RUNNER_FILES == expected_runner_files
    for (domain, runner_file) in expected_runner_files
        @test ActionGradientMCTSExperiments.runner_file(domain) == runner_file
        @test isfile(joinpath(dirname(pathof(ActionGradientMCTSExperiments)), "runners", runner_file))
    end

    @test_throws ArgumentError ActionGradientMCTSExperiments.runner_file("unknown-domain")

    @test isdefined(ActionGradientMCTSExperiments, :VOOSampling)
    @test isdefined(ActionGradientMCTSExperiments, :ProbMountainCar)
    @test isdefined(ActionGradientMCTSExperiments, :LunarLander)
    @test isdefined(ActionGradientMCTSExperiments, :CollabLightDark)
end

@testset "run_experiment solver selection" begin
    base_args = ["--domain", "lunar-lander", "--mode", "ce-opt"]

    default_args = parsed_runner_args(base_args)
    @test default_args["solvers"] == ["ag-dpw"]
    @test !haskey(default_args, "voo")
    @test !haskey(default_args, "pftdpw")
    @test !haskey(default_args, "agmcts")
    @test !haskey(default_args, "pomcpow")

    subset_args = parsed_runner_args([base_args; "--solver"; "dpw"; "ag-dpw"; "vomcpow"])
    @test subset_args["solvers"] == ["dpw", "ag-dpw", "vomcpow"]

    deduped_args = parsed_runner_args([base_args; "--solver"; "dpw"; "dpw"; "ag-vpw"])
    @test deduped_args["solvers"] == ["dpw", "ag-vpw"]

    all_pomdp_args = parsed_runner_args([base_args; "--solver"; "all"])
    @test all_pomdp_args["solvers"] == ["dpw", "ag-dpw", "vpw", "ag-vpw", "pomcpow", "vomcpow"]

    all_mdp_args = parsed_runner_args([base_args; "--solver"; "all"; "--mdp"])
    @test all_mdp_args["solvers"] == ["dpw", "ag-dpw", "vpw", "ag-vpw"]

    @test_throws ArgumentError parsed_runner_args([base_args; "--solver"; "pomcpow"; "--mdp"])
    @test_throws ArgumentError parsed_runner_args([base_args; "--solver"; "vomcpow"; "--mdp"])
    @test_throws ArgumentError parsed_runner_args([base_args; "--solver"; "all"; "dpw"])
    @test_throws ArgParseError parsed_commandline([base_args; "--solver"; "agmcts"])
    @test_throws ArgParseError parsed_commandline([base_args; "--solver"; "pftdpw"])
    @test_throws ArgParseError parsed_commandline([base_args; "--solver"; "ag-dpw"; "--voo"])
end

@testset "solver keyword helpers" begin
    @test ActionGradientMCTSExperiments.normalize_solver_selection(["all"]; mdp=false) == ["dpw", "ag-dpw", "vpw", "ag-vpw", "pomcpow", "vomcpow"]
    @test ActionGradientMCTSExperiments.normalize_solver_selection(["all"]; mdp=true) == ["dpw", "ag-dpw", "vpw", "ag-vpw"]
    @test ActionGradientMCTSExperiments.solver_uses_voo("vpw")
    @test ActionGradientMCTSExperiments.solver_uses_voo("ag-vpw")
    @test ActionGradientMCTSExperiments.solver_uses_voo("vomcpow")
    @test !ActionGradientMCTSExperiments.solver_uses_voo("dpw")
    @test ActionGradientMCTSExperiments.canonical_solver_name("dpw"; mdp=true) == "DPW"
    @test ActionGradientMCTSExperiments.canonical_solver_name("dpw"; mdp=false) == "PFT-DPW"
    @test ActionGradientMCTSExperiments.canonical_solver_name("ag-dpw"; mdp=true) == "AG-DPW"
    @test ActionGradientMCTSExperiments.canonical_solver_name("ag-dpw"; mdp=false) == "AG-PFT-DPW"
    @test ActionGradientMCTSExperiments.canonical_solver_name("vpw"; mdp=true) == "VPW"
    @test ActionGradientMCTSExperiments.canonical_solver_name("vpw"; mdp=false) == "PFT-VPW"
    @test ActionGradientMCTSExperiments.canonical_solver_name("ag-vpw"; mdp=true) == "AG-VPW"
    @test ActionGradientMCTSExperiments.canonical_solver_name("ag-vpw"; mdp=false) == "AG-PFT-VPW"
    @test ActionGradientMCTSExperiments.canonical_solver_name("pomcpow"; mdp=false) == "POMCPOW"
    @test ActionGradientMCTSExperiments.canonical_solver_name("vomcpow"; mdp=false) == "VOMCPOW"
end
