## Heuristics

import BasicPOMCP: SolvedFORollout
import MCTS: next_action, n_children, StateNode, DPWStateNode, action, children, SolvedRolloutEstimator
import POMDPTools.ModelTools: UnderlyingMDP
using ForwardDiff
using Enzyme

mutable struct LanderSolver <: POMDPs.Solver
end
POMDPs.solve(s::LanderSolver, pp::LunarLanderProblem) = LanderPolicy(pp)

## GenerativeBeliefMDP

POMDPs.solve(s::LanderSolver, up::GenerativeBeliefMDP) = POMDPs.solve(s, up.pomdp)
POMDPs.solve(s::LanderSolver, up::GenerativeBeliefPropMDP) = POMDPs.solve(s, up.gmdp.pomdp)

POMDPs.reward(p::GenerativeBeliefMDP{P,U,T,B,A}, s::Vec6, a::Vec3, sp::Vec6) where {P<:LunarLanderProblem,U,T,B,A} = reward(p.pomdp, s, a, sp)
POMDPs.reward(p::GenerativeBeliefPropMDP{P,U,T,B,A}, s::Vec6, a::Vec3, sp::Vec6) where {P<:LunarLanderProblem,U,T,B,A} = reward(p.gmdp.pomdp, s, a, sp)

POMDPs.isterminal(p::GenerativeBeliefMDP{P,U,T,B,A}, s::Vec6) where {P<:LunarLanderProblem,U,T,B,A} = isterminal(p.pomdp, s)
POMDPs.isterminal(p::GenerativeBeliefPropMDP{P,U,T,B,A}, s::Vec6) where {P<:LunarLanderProblem,U,T,B,A} = isterminal(p.gmdp.pomdp, s)
POMDPs.isterminal(p::GenerativeBeliefMDP{P,U,T,B,A}, b::AbstractParticleBelief{Vec6}) where {P<:LunarLanderProblem,U<:BasicParticleFilter,T,B,A} = all(isterminal(p.pomdp, s) for s in particles(b))
POMDPs.isterminal(p::GenerativeBeliefPropMDP{P,U,T,B,A}, b::AbstractParticleBelief{Vec6}) where {P<:LunarLanderProblem,U<:BasicParticleFilter,T,B,A} = all(isterminal(p.gmdp.pomdp, s) for s in particles(b))

## Gradients and transition probabilities - for value gradients

function ActionGradientMCTS.grad_reward(m::LunarLanderProblem, s::Vec6, a::Vec3, sp::Vec6)
    ret = Enzyme.gradient(Forward, reward, Const(m), Const(s), a, Const(sp))
    return ret[3]
end
ActionGradientMCTS.grad_reward(m::GenerativeBeliefMDP{P,U,T,B,A}, s::Vec6, a::Vec3, sp::Vec6) where {P<:LunarLanderProblem,U,T,B,A} = grad_reward(m.pomdp, s, a, sp)
ActionGradientMCTS.grad_reward(m::GenerativeBeliefPropMDP{P,U,T,B,A}, s::Vec6, a::Vec3, sp::Vec6) where {P<:LunarLanderProblem,U,T,B,A} = grad_reward(m.gmdp.pomdp, s, a, sp)

function ActionGradientMCTS.grad_reward(m::GenerativeBeliefPropMDP{P,U,T,B,A},
    s::AbstractParticleBelief{Vec6},
    a::Vec3,
    sp::AbstractParticleBelief{Vec6}) where {P<:LunarLanderProblem,U,T,B,A}
    return @views reduce(+, map((i) -> grad_reward(m.gmdp.pomdp, particle(s, i), a, particle(sp, i)), 1:n_particles(s)))
end

ActionGradientMCTS.transition_log_likelihood(m::LunarLanderProblem, s::Vec6, a, sp::Vec6) = LunarLander.transition_logpdf(m, s, a, sp)

function ActionGradientMCTS.transition_log_likelihood(m::GenerativeBeliefPropMDP{P,U,T,B,A},
    s::Vec6, a::Vec3, sp::Vec6) where {P<:LunarLanderProblem,U,T,B,A}
    return transition_log_likelihood(m.gmdp.pomdp, s, a, sp)
end

function ActionGradientMCTS.transition_log_likelihood(m::GenerativeBeliefPropMDP{P,U,T,B,A},
    s::AbstractParticleBelief{Vec6},
    a::Vec3,
    sp::AbstractParticleBelief{Vec6}) where {P<:LunarLanderProblem,U,T,B,A}
    return @views reduce(+, map((i) -> transition_log_likelihood(m.gmdp.pomdp, particle(s, i), a, particle(sp, i)), 1:n_particles(s)))
end

function ActionGradientMCTS.grad_log_transition(m::LunarLanderProblem, s::Vec6, a::Vec3, sp::Vec6)
    faster_grad = LunarLander.transition_gradlogpdf(m, s, a, sp)
    return faster_grad
end

function ActionGradientMCTS.grad_log_transition(m::GenerativeBeliefPropMDP{P,U,T,B,A},
    s::Vec6, a::Vec3, sp::Vec6) where {P<:LunarLanderProblem,U,T,B,A}
    return grad_log_transition(m.gmdp.pomdp, s, a, sp)
end

function ActionGradientMCTS.grad_log_transition(m::GenerativeBeliefPropMDP{P,U,T,B,A}, s::AbstractParticleBelief{Vec6}, a::Vec3, sp::AbstractParticleBelief{Vec6}) where {P<:LunarLanderProblem,U,T,B,A}
    return reduce(+, map((i) -> grad_log_transition(m.gmdp.pomdp, particle(s, i), a, particle(sp, i)), 1:n_particles(s)))
end

# Projection functions


ActionGradientMCTS.project_action(p::LunarLanderProblem, a::Vec3) = LunarLander._project_action(UnderlyingMDP(p), a)
ActionGradientMCTS.project_action(p::GenerativeBeliefMDP{P,U,T,B,A}, a::Vec3) where {P<:LunarLanderProblem,U,T,B,A} = ActionGradientMCTS.project_action(p.pomdp, a)
ActionGradientMCTS.project_action(p::GenerativeBeliefPropMDP{P,U,T,B,A}, a::Vec3) where {P<:LunarLanderProblem,U,T,B,A} = ActionGradientMCTS.project_action(p.gmdp.pomdp, a)

VOOSampling.project_action(p::Union{
        P,
        GenerativeBeliefMDP{P,U,T,B,A},
        GenerativeBeliefPropMDP{P,U,T,B,A}
    },
    a::Vec3) where {P<:LunarLanderProblem,U,T,B,A} = ActionGradientMCTS.project_action(p, a)
