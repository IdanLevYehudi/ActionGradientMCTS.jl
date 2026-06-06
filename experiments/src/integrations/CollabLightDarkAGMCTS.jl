import BasicPOMCP: SolvedFORollout
import MCTS: next_action, n_children, StateNode, DPWStateNode, action, children, SolvedRolloutEstimator
import POMDPTools.ModelTools: UnderlyingMDP
# using ForwardDiff
using Enzyme
using Zygote

## Action generation

function prod_cardinal_directions(D, K, max_a)
    list = []
    for t in Iterators.product(Iterators.repeated(cardinal_directions(D, max_a; type_func=SVector{D,Float64}), K)...)
        push!(list, SVector{D * K,Float64}(vcat(t...)))
    end
    return list
end

## GenerativeBeliefMDP

ActionGradientMCTS.project_action(pp::CollabLightDarkProblem{D,K,N}, a::SVector{N,Float64}) where {D,K,N} = CollabLightDark._project_action(UnderlyingMDP(pp), a)
ActionGradientMCTS.project_action(p::GenerativeBeliefMDP{P,U,T,B,A}, a::SVector{N,Float64}) where {D,K,N,P<:CollabLightDarkProblem{D,K,N},U,T,B,A} = CollabLightDark._project_action(p.pomdp, a)
ActionGradientMCTS.project_action(p::GenerativeBeliefPropMDP{P,U,T,B,A}, a::SVector{N,Float64}) where {D,K,N,P<:CollabLightDarkProblem{D,K,N},U,T,B,A} = CollabLightDark._project_action(p.gmdp.pomdp, a)

VOOSampling.project_action(p::Union{
        P,
        GenerativeBeliefMDP{P,U,T,B,A},
        GenerativeBeliefPropMDP{P,U,T,B,A}
    },
    a::SVector{N,Float64}) where {D,K,N,P<:CollabLightDarkProblem{D,K,N},U,T,B,A} = ActionGradientMCTS.project_action(p, a)


POMDPs.solve(s::CLDStraightToGoalSolver, up::GenerativeBeliefMDP) = POMDPs.solve(s, up.pomdp)
POMDPs.solve(s::CLDStraightToGoalSolver, up::GenerativeBeliefPropMDP) = POMDPs.solve(s, up.gmdp.pomdp)

POMDPs.reward(p::GenerativeBeliefMDP{P,U,T,B,A}, s::SVector{N,Float64}, a::SVector{N,Float64}, sp::SVector{N,Float64}) where {D,K,N,P<:CollabLightDarkProblem{D,K,N},U,T,B,A} = reward(p.pomdp, s, a, sp)
POMDPs.reward(p::GenerativeBeliefPropMDP{P,U,T,B,A}, s::SVector{N,Float64}, a::SVector{N,Float64}, sp::SVector{N,Float64}) where {D,K,N,P<:CollabLightDarkProblem{D,K,N},U,T,B,A} = reward(p.gmdp.pomdp, s, a, sp)

POMDPs.isterminal(p::GenerativeBeliefMDP{P,U,T,B,A}, s::SVector{N,Float64}) where {D,K,N,P<:CollabLightDarkProblem{D,K,N},U,T,B,A} = isterminal(p.pomdp, s)
POMDPs.isterminal(p::GenerativeBeliefPropMDP{P,U,T,B,A}, s::SVector{N,Float64}) where {D,K,N,P<:CollabLightDarkProblem{D,K,N},U,T,B,A} = isterminal(p.gmdp.pomdp, s)
POMDPs.isterminal(p::GenerativeBeliefMDP{P,U,T,B,A}, b::AbstractParticleBelief{SVector{N,Float64}}) where {D,K,N,P<:CollabLightDarkProblem{D,K,N},U<:BasicParticleFilter,T,B,A} = isterminal(p.pomdp, b)
POMDPs.isterminal(p::GenerativeBeliefPropMDP{P,U,T,B,A}, b::AbstractParticleBelief{SVector{N,Float64}}) where {D,K,N,P<:CollabLightDarkProblem{D,K,N},U<:BasicParticleFilter,T,B,A} = isterminal(p.gmdp.pomdp, b)

## Gradients and transition probabilities - for value gradients

function ActionGradientMCTS.grad_reward(m::CollabLightDarkProblem{D,K,N}, s::SVector{N,Float64}, a::SVector{N,Float64}, sp::SVector{N,Float64}) where {D,K,N}
    try
        return CollabLightDark._grad_reward(m, s, a, sp)
    catch ex
        @warn "Caught exception during grad_reward" maxlog = 1
        @warn ex maxlog = 1
        t = first(Zygote.gradient((a) -> reward(m, s, a, sp), a))
        if t === nothing
            return zero(a)
        else
            return t
        end
    end
end

ActionGradientMCTS.grad_reward(m::GenerativeBeliefMDP{P,U,T,B,A}, s::SVector{N,Float64}, a::SVector{N,Float64}, sp::SVector{N,Float64}) where {D,K,N,P<:CollabLightDarkProblem{D,K,N},U,T,B,A} = grad_reward(m.pomdp, s, a, sp)
ActionGradientMCTS.grad_reward(m::GenerativeBeliefPropMDP{P,U,T,B,A}, s::SVector{N,Float64}, a::SVector{N,Float64}, sp::SVector{N,Float64}) where {D,K,N,P<:CollabLightDarkProblem{D,K,N},U,T,B,A} = grad_reward(m.gmdp.pomdp, s, a, sp)

function ActionGradientMCTS.grad_reward(m::GenerativeBeliefPropMDP{P,U,T,B,A}, s::AbstractParticleBelief{SVector{N,Float64}}, a::SVector{N,Float64}, sp::AbstractParticleBelief{SVector{N,Float64}}) where {D,K,N,P<:CollabLightDarkProblem{D,K,N},U,T,B,A}
    return @views reduce(+, map((i) -> grad_reward(m.gmdp.pomdp, particle(s, i), a, particle(sp, i)) * weight(s, i), 1:n_particles(s))) / weight_sum(s)
end

function ActionGradientMCTS.transition_log_likelihood(m::CollabLightDarkProblem{D,K,N}, s::SVector{N,Float64}, a::SVector{N,Float64}, sp::SVector{N,Float64}) where {D,K,N}
    m = UnderlyingMDP(m)
    return logpdf(transition(m, s, a), sp)
end

function ActionGradientMCTS.transition_log_likelihood(m::GenerativeBeliefPropMDP{P,U,T,B,A},
    s::SVector{N,Float64}, a::SVector{N,Float64}, sp::SVector{N,Float64}) where {D,K,N,P<:CollabLightDarkProblem{D,K,N},U,T,B,A}
    return transition_log_likelihood(m.gmdp.pomdp, s, a, sp)
end

function ActionGradientMCTS.transition_log_likelihood(m::GenerativeBeliefPropMDP{P,U,T,B,A},
    s::AbstractParticleBelief{SVector{N,Float64}},
    a::SVector{N,Float64},
    sp::AbstractParticleBelief{SVector{N,Float64}}) where {D,K,N,P<:CollabLightDarkProblem{D,K,N},U,T,B,A}
    # Tested to be the fastest and with least allocations!!
    # This beat the following:
    # mapreduce, sum, explicit loop
    # Tested all with/without: @inbounds, @views
    sum = @views reduce(+, map((i) -> transition_log_likelihood(m.gmdp.pomdp, particle(s, i), a, particle(sp, i)), 1:n_particles(s)))
    return sum
end

# Calculates the propagated belief likelihood for the unordered, unweighted particle belief space.
function transition_log_likelihood2(m::GenerativeBeliefPropMDP{P,U,T,B,A},
    s::AbstractParticleBelief{SVector{N,Float64}},
    a::SVector{N,Float64},
    sp::AbstractParticleBelief{SVector{N,Float64}}) where {D,K,N,P<:CollabLightDarkProblem{D,K,N},U,T,B,A}
    sum = 0.0
    for j in 1:n_particles(sp)
        mid_sum = 0.0
        sum += logsumexp(transition_log_likelihood(m.gmdp.pomdp, particle(s, i), a, particle(sp, j)) for i in 1:n_particles(s))
    end
    sum -= n_particles(sp) * log(n_particles(s))
end

function ActionGradientMCTS.grad_log_transition(m::CollabLightDarkProblem{D,K,N}, s::SVector{N,Float64}, a::SVector{N,Float64}, sp::SVector{N,Float64}) where {D,K,N}
    try
        t = CollabLightDark._grad_log_transition(m, s, a, sp)
        return t
    catch ex
        @warn "Caught exception during grad_log_transition" maxlog = 1
        @warn ex maxlog = 1
        t = first(Zygote.gradient((a) -> transition_log_likelihood(m, s, a, sp), a))
        if t === nothing
            return zero(a)
        else
            return t
        end
    end
end

function ActionGradientMCTS.grad_log_transition(m::GenerativeBeliefPropMDP{P,U,T,B,A},
    s::SVector{N,Float64}, a::SVector{N,Float64}, sp::SVector{N,Float64}) where {D,K,N,P<:CollabLightDarkProblem{D,K,N},U,T,B,A}
    return grad_log_transition(m.gmdp.pomdp, s, a, sp)
end

function ActionGradientMCTS.grad_log_transition(m::GenerativeBeliefPropMDP{P,U,T,B,A}, s::AbstractParticleBelief{SVector{N,Float64}}, a::SVector{N,Float64}, sp::AbstractParticleBelief{SVector{N,Float64}}) where {D,K,N,P<:CollabLightDarkProblem{D,K,N},U,T,B,A}
    # This explicit loop was the fastest from all the tested variations
    # naive sum, mapreduce(), map(+, reduce()), with/without @inbounds, @views
    sum = MVector{length(SVector{N,Float64}),Float64}(zero(SVector{N,Float64}))
    @inbounds @views for i in 1:n_particles(s)
        sum += grad_log_transition(m.gmdp.pomdp, particle(s, i), a, particle(sp, i))
    end
    return SVector{N,Float64}(sum)
end

function ActionGradientMCTS.grad_transition_likelihood(m::GenerativeBeliefPropMDP{P,U,T,B,A}, s::AbstractParticleBelief{SVector{N,Float64}}, a::SVector{N,Float64}, sp::AbstractParticleBelief{SVector{N,Float64}}) where {D,K,N,P<:CollabLightDarkProblem{D,K,N},U,T,B,A}
    return exp(transition_log_likelihood(m, s, a, sp)) * grad_log_transition(m, s, a, sp)
end

## Naive estimate value?

# Calculate esitmate value based on:
# 1 / Belief cov det
# det(cov) > (transition_cov) ^ N starts to become not good.
# Distance of belief mean to goal - Needs to reach goal by CLDStraightToGoalPolicy within steps
function estimate_value_distance_based(pp::CollabLightDarkProblem{D,K,N}, s::SVector{N,Float64}, steps::Int) where {D,K,N}
    if isterminal(pp, s)
        return 0.0
    end
    policy = CLDStraightToGoalPolicy(pp, Random.default_rng())  # rng is not used for states so it doesn't matter
    mdp = UnderlyingMDP(pp)
    disc = 1.0
    r_total = 0.0
    step = 1
    while !isterminal(mdp, s) && step <= steps
        a = action(policy, s)
        sp = s .+ a
        r = reward(mdp, s, a, sp)
        r_total += disc * r
        s = sp
        disc *= discount(mdp)
        step += 1
    end
    return r_total
end

function estimate_value_distance_based(pp::CollabLightDarkProblem{D,K,N}, b::AbstractParticleBelief{SVector{N,Float64}}, steps::Int) where {D,K,N}
    if isterminal(pp, b)
        return 0.0
    end
    m = UnderlyingMDP(pp)
    cov = Statistics.cov(b)
    if isfinite(det(cov))
        # Volume of ellipse of n stds is n*π*sqrt(det(cov)), here we take n=3.
        # https://math.stackexchange.com/questions/2751632/solve-for-volume-of-ellipsoid-mathbb-x-mathbf-mut-sigma-1-mathbb-x
        cov_volume_prop = max((m.goal_area^K) / 3 * π * sqrt(max(0, det(cov))), 1)  # We don't want to scale the returned reward by more than 1. Taking the sqrt because the det is std^2.
    else
        cov_volume_prop = 1
    end
    s = mean(b)  # This could be replaced with a random state
    r_total = estimate_value_distance_based(m, s, steps)
    return r_total / cov_volume_prop
end

estimate_value_distance_based(pp::CollabLightDarkProblem{D,K,N}, b::POMDPTools.TerminalState, steps::Int) where {D,K,N} = 0.0

estimate_value_distance_based(m::GenerativeBeliefPropMDP{P,U,T,B,A}, b::B, steps::Int) where {D,K,N,P<:CollabLightDarkProblem{D,K,N},U,T,B,A} = estimate_value_distance_based(m.gmdp.pomdp, b, steps)
estimate_value_distance_based(m::GenerativeBeliefMDP{P,U,T,B,A}, b::B, steps::Int) where {D,K,N,P<:CollabLightDarkProblem{D,K,N},U,T,B,A} = estimate_value_distance_based(m.pomdp, b, steps)

estimate_value_distance_based(pp::CollabLightDarkProblem{D,K,N}, start_state, h::BeliefNode, steps::Int) where {D,K,N} = estimate_value_distance_based(pp, start_state, steps)
estimate_value_distance_based(m::GenerativeBeliefPropMDP{P,U,T,B,A}, start_state, h::BeliefNode, steps::Int) where {D,K,N,P<:CollabLightDarkProblem{D,K,N},U,T,B,A} = estimate_value_distance_based(m.gmdp.pomdp, start_state, steps)
estimate_value_distance_based(m::GenerativeBeliefMDP{P,U,T,B,A}, start_state, h::BeliefNode, steps::Int) where {D,K,N,P<:CollabLightDarkProblem{D,K,N},U,T,B,A} = estimate_value_distance_based(m.pomdp, start_state, steps)
