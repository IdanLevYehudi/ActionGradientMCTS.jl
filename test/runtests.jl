using Test
using ActionGradientMCTS
using TOML

const REPO_ROOT = dirname(@__DIR__)
const SRC_DIR = joinpath(REPO_ROOT, "src")
const EXPERIMENTS_DIR = joinpath(REPO_ROOT, "experiments")

@testset "ActionGradientMCTS public repo" begin
    @test isfile(joinpath(REPO_ROOT, "Project.toml"))
    @test isfile(joinpath(SRC_DIR, "ActionGradientMCTS.jl"))

    @test isdefined(ActionGradientMCTS, :ActionGradMCTSSolver)
    @test isdefined(ActionGradientMCTS, :ActionGradMCTSPlanner)
    @test isdefined(ActionGradientMCTS, :GenerativeBeliefPropMDP)
    @test isdefined(ActionGradientMCTS, :grad_reward)
    @test isdefined(ActionGradientMCTS, :grad_log_transition)
    @test isdefined(ActionGradientMCTS, :transition_log_likelihood)
    @test isdefined(ActionGradientMCTS, :project_action)
    @test !isdefined(ActionGradientMCTS, :ActionGradMCTS)
    @test !isdefined(ActionGradientMCTS, :VOOTreeSearch)
end

@testset "Package and experiment boundaries" begin
    project = TOML.parsefile(joinpath(REPO_ROOT, "Project.toml"))
    root_deps = keys(project["deps"])
    @test !("ArgParse" in root_deps)
    @test !("BenchmarkTools" in root_deps)
    @test !("CSV" in root_deps)
    @test !("D3Trees" in root_deps)
    @test !("Plots" in root_deps)
    @test !("POMDPGifs" in root_deps)

    @test isfile(joinpath(EXPERIMENTS_DIR, "Project.toml"))
    @test isfile(joinpath(EXPERIMENTS_DIR, "run_experiment.jl"))
    @test isfile(joinpath(EXPERIMENTS_DIR, "src", "ActionGradientMCTSExperiments.jl"))
    @test isfile(joinpath(EXPERIMENTS_DIR, "src", "baselines", "VOOSampling.jl"))
    @test !isfile(joinpath(REPO_ROOT, "scripts", "MainMountainCar.jl"))
    @test !isfile(joinpath(REPO_ROOT, "scripts", "MainLunarLander.jl"))
    @test !isfile(joinpath(REPO_ROOT, "scripts", "MainCollabLightDark.jl"))
end
