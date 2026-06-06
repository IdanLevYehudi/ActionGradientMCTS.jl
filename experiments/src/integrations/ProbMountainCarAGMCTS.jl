## Heuristics

import BasicPOMCP: SolvedFORollout
import MCTS: next_action, n_children, StateNode, DPWStateNode, action, children, SolvedRolloutEstimator
import POMDPTools.ModelTools: UnderlyingMDP
using ForwardDiff
using Enzyme

@with_kw mutable struct ThresholdVPolicy <: Policy
    prob::ProbMountainCarProblem
    threshold::Float64 = 0.0
    rng::Union{TaskLocalRNG,AbstractRNG} = Random.GLOBAL_RNG
end

POMDPs.action(p::ThresholdVPolicy, s::CarState) = s[2] < p.threshold ? UnderlyingMDP(p.prob).action_min : UnderlyingMDP(p.prob).action_max
POMDPs.action(p::ThresholdVPolicy, b::ParticleFilters.AbstractParticleBelief{CarState}) = action(p, rand(p.rng, b))

@with_kw mutable struct ThresholdVSolver <: POMDPs.Solver
    threshold::Float64 = 0.0
    rng::Union{TaskLocalRNG,AbstractRNG} = Random.GLOBAL_RNG
end
POMDPs.solve(s::ThresholdVSolver, pp::ProbMountainCarProblem) = ThresholdVPolicy(prob=UnderlyingMDP(pp), threshold=s.threshold, rng=s.rng)

## GenerativeBeliefMDP

POMDPs.solve(s::ThresholdVSolver, up::GenerativeBeliefMDP) = POMDPs.solve(s, up.pomdp)
POMDPs.solve(s::ThresholdVSolver, up::GenerativeBeliefPropMDP) = POMDPs.solve(s, up.gmdp.pomdp)

mutable struct MaxActionsFirstGen
    rng::AbstractRNG
end

function MCTS.next_action(gen::MaxActionsFirstGen, pp::ProbMountainCarPOMDP, b, node)
    p = UnderlyingMDP(pp)
    if n_children(node) < 1
        a = rand(gen.rng, [p.action_max, p.action_min])
    elseif n_children(node) < 2
        prev_action = child_action_labels(node)[1]  # There should be only a single action
        a = prev_action == p.action_max ? p.action_min : p.action_max
    else
        a = rand(gen.rng, actions(p))
    end
    return a
end

MCTS.next_action(gen::MaxActionsFirstGen, p::GenerativeBeliefMDP, b, node) = next_action(gen, p.pomdp, b, node)
MCTS.next_action(gen::MaxActionsFirstGen, p::GenerativeBeliefPropMDP, b, node) = next_action(gen, p.gmdp.pomdp, b, node)

POMDPs.reward(p::GenerativeBeliefMDP{P,U,T,B,A}, s::CarState, a, sp::CarState) where {P<:ProbMountainCarProblem,U,T,B,A} = reward(p.pomdp, s, a, sp)
POMDPs.reward(p::GenerativeBeliefPropMDP{P,U,T,B,A}, s::CarState, a, sp::CarState) where {P<:ProbMountainCarProblem,U,T,B,A} = reward(p.gmdp.pomdp, s, a, sp)

POMDPs.isterminal(p::GenerativeBeliefMDP{P,U,T,B,A}, s::CarState) where {P<:ProbMountainCarProblem,U,T,B,A} = isterminal(p.pomdp, s)
POMDPs.isterminal(p::GenerativeBeliefPropMDP{P,U,T,B,A}, s::CarState) where {P<:ProbMountainCarProblem,U,T,B,A} = isterminal(p.gmdp.pomdp, s)
POMDPs.isterminal(p::GenerativeBeliefMDP{P,U,T,B,A}, b::AbstractParticleBelief{CarState}) where {P<:ProbMountainCarProblem,U<:BasicParticleFilter,T,B,A} = all(isterminal(p.pomdp, s) for s in particles(b))
POMDPs.isterminal(p::GenerativeBeliefPropMDP{P,U,T,B,A}, b::AbstractParticleBelief{CarState}) where {P<:ProbMountainCarProblem,U<:BasicParticleFilter,T,B,A} = all(isterminal(p.gmdp.pomdp, s) for s in particles(b))

## Gradients and transition probabilities - for value gradients

ActionGradientMCTS.grad_reward(m::ProbMountainCarProblem, s::CarState, a::Float64, sp::CarState) = 2 * UnderlyingMDP(m).action_penalty_coeff * a
ActionGradientMCTS.grad_reward(m::GenerativeBeliefMDP{P,U,T,B,A}, s::CarState, a::Float64, sp::CarState) where {P<:ProbMountainCarProblem,U,T,B,A} = grad_reward(m.pomdp, s, a, sp)
ActionGradientMCTS.grad_reward(m::GenerativeBeliefPropMDP{P,U,T,B,A}, s::CarState, a::Float64, sp::CarState) where {P<:ProbMountainCarProblem,U,T,B,A} = grad_reward(m.gmdp.pomdp, s, a, sp)

function ActionGradientMCTS.grad_reward(m::GenerativeBeliefPropMDP{P,U,T,B,A}, s::AbstractParticleBelief{CarState}, a::Float64, sp::AbstractParticleBelief{CarState}) where {P<:ProbMountainCarProblem,U,T,B,A}
    return @views reduce(+, map((i) -> grad_reward(m.gmdp.pomdp, particle(s, i), a, particle(sp, i)) * weight(s, i), 1:n_particles(s))) / weight_sum(s)
end

ActionGradientMCTS.transition_log_likelihood(m::ProbMountainCarProblem, s::CarState, a, sp::CarState) = ProbMountainCar.transition_logpdf(m, s, a, sp)

function ActionGradientMCTS.transition_log_likelihood(m::GenerativeBeliefPropMDP{P,U,T,B,A},
    s::CarState, a, sp::CarState) where {P<:ProbMountainCarProblem,U,T,B,A}
    return transition_log_likelihood(m.gmdp.pomdp, s, a, sp)
end

function ActionGradientMCTS.transition_log_likelihood(m::GenerativeBeliefPropMDP{P,U,T,B,A},
    s::AbstractParticleBelief{CarState},
    a,
    sp::AbstractParticleBelief{CarState}) where {P<:ProbMountainCarProblem,U,T,B,A}
    return @views reduce(+, map((i) -> transition_log_likelihood(m.gmdp.pomdp, particle(s, i), a, particle(sp, i)), 1:n_particles(s)))
end

function ActionGradientMCTS.grad_log_transition(m::ProbMountainCarProblem, s::CarState, a::Float64, sp::CarState)
    return ProbMountainCar.transition_gradlogpdf(m, s, a, sp)
end

function ActionGradientMCTS.grad_log_transition(m::GenerativeBeliefPropMDP{P,U,T,B,A},
    s::CarState, a::Float64, sp::CarState) where {P<:ProbMountainCarProblem,U,T,B,A}
    return grad_log_transition(m.gmdp.pomdp, s, a, sp)
end

function ActionGradientMCTS.grad_log_transition(m::GenerativeBeliefPropMDP{P,U,T,B,A}, s::AbstractParticleBelief{CarState}, a::Float64, sp::AbstractParticleBelief{CarState}) where {P<:ProbMountainCarProblem,U,T,B,A}
    return reduce(+, map((i) -> grad_log_transition(m.gmdp.pomdp, particle(s, i), a, particle(sp, i)), 1:n_particles(s)))
end

ActionGradientMCTS.project_action(p::ProbMountainCarProblem, a::Float64) = ProbMountainCar.project_action(p, a)
ActionGradientMCTS.project_action(p::GenerativeBeliefMDP{P,U,T,B,A}, a::Float64) where {P<:ProbMountainCarProblem,U,T,B,A} = ProbMountainCar.project_action(p.pomdp, a)
ActionGradientMCTS.project_action(p::GenerativeBeliefPropMDP{P,U,T,B,A}, a::Float64) where {P<:ProbMountainCarProblem,U,T,B,A} = ProbMountainCar.project_action(p.gmdp.pomdp, a)

VOOSampling.project_action(p::Union{
        P,
        GenerativeBeliefMDP{P,U,T,B,A},
        GenerativeBeliefPropMDP{P,U,T,B,A}
    },
    a::Float64) where {P<:ProbMountainCarProblem,U,T,B,A} = ActionGradientMCTS.project_action(p, a)
