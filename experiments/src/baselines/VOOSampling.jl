module VOOSampling

using POMDPs
using BasicPOMCP
using POMCPOW
using ParticleFilters
using Parameters
using MCTS
using Random
using Distributions
using LinearAlgebra
using StaticArrays
using MCTS

using ActionGradientMCTS

import MCTS: next_action

export
    VOOActionGenerator,
    child_action_labels,
    voo_sample,
    project_action

###
# VOO.jl contents
# Copied here due to Revise.jl issues...
###

struct VOOActionGenerator{D1<:Distribution,D2<:Distribution,R<:AbstractRNG} <: Function
    exploration_prob::Float64
    exploration_sampler::D1
    voronoi_sampler::D2
    action_rng::R
    acceptance_radius_sq::Float64
    early_halt::Bool
    halt_count::Int64
end

# If only using acceptance radius
function VOOActionGenerator(exploration_prob::Float64,
    exploration_sampler::Distribution,
    voronoi_sampler::Distribution,
    action_rng::AbstractRNG,
    acceptance_radius_sq::Float64)
    early_halt = false
    halt_count = 1
    return VOOActionGenerator(exploration_prob, exploration_sampler, voronoi_sampler, action_rng, acceptance_radius_sq, early_halt, halt_count)
end

# If only using halt count
function VOOActionGenerator(exploration_prob::Float64,
    exploration_sampler::Distribution,
    voronoi_sampler::Distribution,
    early_halt::Bool,
    halt_count::Int64)
    acceptance_radius_sq = 0.0
    return VOOActionGenerator(exploration_prob, exploration_sampler, voronoi_sampler, action_rng, acceptance_radius_sq, early_halt, halt_count)
end

function MCTS.next_action(f::VOOActionGenerator, a_list, q_list, a_space, project_func::Function=Identity)
    # determine explore vs. exploit
    w = rand(f.exploration_sampler)

    if w <= f.exploration_prob || length(a_list) <= 1
        # uniform sample from action space
        a_new = rand(f.action_rng, a_space)
    else
        # find the best value
        a_new = voronoi_sample_centered(f, a_list, q_list, project_func)
    end
    return a_new
end

function project_action end
function child_action_labels end
function child_q_values end

# function project_action(problem::Union{MDP,POMDP}, a)
#     return a
# end


function MCTS.next_action(f::VOOActionGenerator, problem::Union{MDP,POMDP}, b, h::N) where N<:Union{ActionGradStateNode, DPWStateNode, StateNode, POWTreeObsNode}
    # extract actions
    # t = h.tree
    # a_inds = t.children[children_indices(h)]
    # a_list = t.a_labels[a_inds]
    a_list = child_action_labels(h)
    a_space = actions(problem, b)
    q_list = child_q_values(h)
    project_func = (a) -> VOOSampling.project_action(problem, a)
    return next_action(f, a_list, q_list, a_space, project_func)
end

# Box actions
function action_dist_sq(a1::Union{Number, AbstractVector}, a2::Union{Number, AbstractVector})
    return sum((a1 .- a2) .^ 2)
end

function action_sample_centered(best_a::Union{Number, AbstractVector}, sampler, project_func::Function=Identity)
    in_bounds = false
    a_new = Nothing

    # make sure new sample is in bounds
    while !in_bounds
        a_type = typeof(best_a)
        a_new = a_type(a_type(rand(sampler)) + best_a)
        projected_a = project_func(a_new)
        in_bounds = projected_a == a_new
    end

    return a_new
end

children_indices(h::DPWStateNode) = h.index
children_indices(h::POWTreeObsNode) = h.node

function voronoi_sample_centered(f::VOOActionGenerator, a_list, q_list, project_func::Function=Identity)
    best_a = a_list[argmax(q_list)]
    a_new = best_a
    closest = false

    a_closest = best_a
    dist_closest_sq = Inf

    # iterate until it lies within the best Voronoi cell
    n = 1
    while .!closest
        closest = true
        a_new = action_sample_centered(best_a, f.voronoi_sampler, project_func)
        dist_to_best_sq = action_dist_sq(a_new, best_a)

        if dist_to_best_sq < dist_closest_sq
            a_closest = a_new
            dist_closest_sq = dist_to_best_sq
        end

        # if early halting, return the closest one to the center
        if f.early_halt && n >= f.halt_count
            return a_closest
        end

        # if auto acceptance radius, return if the sample is within the radius
        if dist_to_best_sq < f.acceptance_radius_sq
            return a_new
        end

        # iterate over all actions to see if it is closer to another Voronoi cell
        for a in a_list
            if action_dist_sq(a_new, a) < dist_to_best_sq
                closest = false
            end
        end
        n += 1
    end
    return a_new
end

function (f::VOOActionGenerator)(pomdp, b, h)
    a_new = next_action(f, pomdp, b, h)
    return a_new
end

end # module VOOSampling
