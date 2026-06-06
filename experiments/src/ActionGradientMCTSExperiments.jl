module ActionGradientMCTSExperiments

using ActionGradientMCTS
using ArgParse
using BasicPOMCP
using BenchmarkTools
using CSV
using CovarianceEstimation
using DataFrames
using Dates
using Distributed
using Distributions
using Enzyme
using Flux
using ForwardDiff
using JSON
using LinearAlgebra
using MCTS
using POMCPOW
using POMDPs
using POMDPTools
using Parameters
using ParticleFilters
using Plots
using ProgressMeter
using Random
using StaticArrays
using Statistics
using StatsBase
using Zygote

import POMDPTools.ModelTools: GenerativeBeliefMDP, UnderlyingMDP

include("baselines/VOOSampling.jl")
include("domains/ProbMountainCar.jl")
include("domains/LunarLander.jl")
include("domains/CollabLightDark.jl")

using .VOOSampling
using .ProbMountainCar
using .LunarLander
using .CollabLightDark

include("integrations/ProbMountainCarAGMCTS.jl")
include("integrations/LunarLanderAGMCTS.jl")
include("integrations/CollabLightDarkAGMCTS.jl")
include("utils/solver_selection.jl")
include("utils/pomdp_utils.jl")
include("utils/pomdp_solver_experiments.jl")

export VOOSampling, ProbMountainCar, LunarLander, CollabLightDark
export VOOActionGenerator
export SimParams, PFTDPWParams, POMCPOWParams, ActionGradMCTSParams
export gen_sims, gather_sim_results, compare_by_max_time_query, cross_entropy_optimization
export MDPSimIterator, POMDPSimIterator
export SOLVER_CHOICES, DEFAULT_SOLVERS, MDP_SOLVERS, POMDP_SOLVERS
export normalize_solver_selection, canonical_solver_name
export solver_uses_voo, solver_is_dpw, solver_is_agmcts, solver_is_pomcpow

const RUNNER_FILES = Dict(
    "mountain-car" => "MountainCarRunner.jl",
    "lunar-lander" => "LunarLanderRunner.jl",
    "collab-light-dark" => "CollabLightDarkRunner.jl",
)

runner_file(domain::AbstractString) = get(RUNNER_FILES, domain) do
    throw(ArgumentError("unknown domain '$domain'. Expected one of: $(join(sort(collect(keys(RUNNER_FILES))), ", "))"))
end

end
