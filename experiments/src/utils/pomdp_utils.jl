using BasicPOMCP: SolvedFORollout
using Statistics
using POMDPTools.Simulators: MDPSimIterator, POMDPSimIterator, out_tuple

### MCTS Related functions

function MCTS.estimate_value(est::SolvedFORollout, mdp::GenerativeBeliefMDP, state, remaining_depth)
    if isterminal(mdp, state)
        return 0.0
    end
    sim = RolloutSimulator(est.rng, remaining_depth)
    return POMDPs.simulate(sim, UnderlyingMDP(mdp.pomdp), est.policy, rand(est.rng, state))
end

MCTS.estimate_value(est::SolvedFORollout, mdp::GenerativeBeliefPropMDP, state, remaining_depth) = estimate_value(est, mdp.gmdp, state, remaining_depth)

# Mean State Rollout
struct MeanStateRollout  # fully observable rollout
    solver::Union{POMDPs.Solver,POMDPs.Policy}
    k::Int  # Number of state samples to estimate the value
end

struct SolvedMeanStateRollout{P<:POMDPs.Policy,M<:Union{MDP,POMDP},RNG<:AbstractRNG}
    policy::P
    m::M
    k::Int
    rng::RNG
end

function MCTS.convert_estimator(ev::MeanStateRollout, solver, pomdp)
    policy = MCTS.convert_to_policy(ev.solver, pomdp)
    return SolvedMeanStateRollout(policy, pomdp, ev.k, solver.rng)
end
MCTS.convert_estimator(ev::MeanStateRollout, solver, pomdp::GenerativeBeliefMDP) = MCTS.convert_estimator(ev, solver, pomdp.pomdp)
MCTS.convert_estimator(ev::MeanStateRollout, solver, pomdp::GenerativeBeliefPropMDP) = MCTS.convert_estimator(ev, solver, pomdp.gmdp.pomdp)

function MCTS.estimate_value(est::SolvedMeanStateRollout, start_state, steps::Int)
    K = est.k
    if start_state isa AbstractParticleBelief
        states = [rand(est.rng, start_state) for _ in 1:K]
    else
        states = [start_state]
        K = 1
    end

    mdp = UnderlyingMDP(est.m)
    disc = 1.0
    r_total = 0.0
    step = 1
    states = [s for s in states if !isterminal(mdp, s)]
    while !(isempty(states)) && step <= steps
        # Clean terminal states
        new_states = []
        mean_state = mean(states)
        a = action(est.policy, mean_state)
        for s in states
            sp, r = @gen(:sp, :r)(mdp, s, a, est.rng)
            r_total += disc * r / K
            if !isterminal(mdp, sp)
                push!(new_states, sp)
            end
        end
        states = new_states
        disc *= discount(mdp)
        step += 1
    end
    return r_total
end

MCTS.estimate_value(est::SolvedMeanStateRollout, b::POMDPTools.TerminalState, steps::Int) = 0.0
MCTS.estimate_value(est::SolvedMeanStateRollout, pomdp::Union{MDP,POMDP}, start_state, steps::Int) = MCTS.estimate_value(est, start_state, steps)
MCTS.estimate_value(est::SolvedMeanStateRollout, pomdp::Union{MDP,POMDP}, start_state, h::BeliefNode, steps::Int) = MCTS.estimate_value(est, start_state, steps)


# An action generator that first produces an action based on a policy (most likely rollout policy) and then proceeds to sample at random.
struct PolicyFirstGen{P<:Policy}
    policy::P
    gen::Any  # Has to have next_action(gen, mdp/pomdp, state, node) function
end

function MCTS.next_action(gen::PolicyFirstGen, p::M, b, node) where {M<:POMDP}
    if n_children(node) < 1
        a = action(gen.policy, mean(b))
    else
        a = next_action(gen.gen, p, b, node)
    end
end

function MCTS.next_action(gen::PolicyFirstGen, p::M, s, node) where {M<:MDP}
    if n_children(node) < 1
        a = action(gen.policy, s)
    else
        a = next_action(gen.gen, p, s, node)
    end
end

child_idx(node::ActionGradStateNode) = [child_id for child_id in children(node)]
child_idx(node::DPWStateNode) = [child_id for child_id in children(node)]
child_idx(node::StateNode) = [child_id for child_id in children(node)]
child_idx(node::POWTreeObsNode) = [child_id for child_id in node.tree.tried[node.node]]

child_action_labels(node::ActionGradStateNode) = [a_labels(node.tree, child_id) for child_id in children(node)]
child_action_labels(node::DPWStateNode) = [node.tree.a_labels[child_id] for child_id in children(node)]
child_action_labels(node::StateNode) = [action(a_node) for a_node in children(node)]
child_action_labels(node::POWTreeObsNode) = [node.tree.a_labels[child_id] for child_id in node.tree.tried[node.node]]

child_q_values(node::ActionGradStateNode) = [q_value(node.tree, child_id) for child_id in children(node)]
child_q_values(node::DPWStateNode) = [node.tree.q[child_id] for child_id in children(node)]
child_q_values(node::StateNode) = [q(child_node) for child_node in children(node)]
child_q_values(node::POWTreeObsNode) = [node.tree.v[child_id] for child_id in node.tree.tried[node.node]]


VOOSampling.child_action_labels(x::Union{ActionGradStateNode, DPWStateNode, StateNode, POWTreeObsNode}) = child_action_labels(x)
VOOSampling.child_q_values(x::Union{ActionGradStateNode, DPWStateNode, StateNode, POWTreeObsNode}) = child_q_values(x)

mutable struct IteratorFirstGen
    it::Any
    gen::Any
    rng::AbstractRNG
    itr_dict::Dict{Any,Any}
    curr_tree_id::UInt64  # Holding a pointer to detect if the tree has changed - if so means that itr_dict needs to be cleared.
end

function IteratorFirstGen(it, gen, rng=Random.GLOBAL_RNG)
    itr_dict = Dict{Any,Any}()
    return IteratorFirstGen(it, gen, rng, itr_dict, 0)
end

index(n::POWTreeObsNode) = n.node
index(n::DPWStateNode) = n.index
index(n::ActionGradStateNode) = n.index

function MCTS.next_action(gen::IteratorFirstGen, p::M, b, node) where {M<:Union{MDP,POMDP}}
    tree_id = objectid(node.tree)
    if tree_id != gen.curr_tree_id
        empty!(gen.itr_dict)
        gen.curr_tree_id = tree_id
    end
    ind = index(node)
    if !haskey(gen.itr_dict, ind)
        rand_it = Random.shuffle(gen.rng, gen.it)
        gen.itr_dict[ind] = Iterators.Stateful(rand_it)
    end
    if isempty(gen.itr_dict[ind])
        return next_action(gen.gen, p, b, node)
    else
        return popfirst!(gen.itr_dict[ind])
    end
end

function cardinal_directions(n, a_max; type_func=nothing)
    list = []
    if type_func === nothing
        type_func = identity
    end
    zeros_n = zeros(n)
    for i in 1:n
        a = deepcopy(zeros_n)
        a[i] = a_max
        push!(list, type_func(a))
        a = deepcopy(zeros_n)
        a[i] = -a_max
        push!(list, type_func(a))
    end
    return list
end

## General functions for particle beliefs

## Calculating cov for particle beliefs of StaticArrays
function Statistics.cov(b::ParticleCollection{T}) where {T<:StaticVector} # uncorrected covariance
    centralized = reduce(hcat, b.particles) .- mean(b)
    centralized * centralized' / length(b.particles) # outer product
end
function Statistics.cov(b::WeightedParticleBelief{T}) where {T<:StaticVector} # uncorrected covariance
    centralized = reduce(hcat, b.particles) .- mean(b)
    (centralized .* b.weights') * centralized' / weight_sum(b)
end


## Fixes for history_recorder.jl in POMDPTools/Simulators
# In finite-horizon POMDPs, the solver should have its max depth adapted to the current simulation horizon.
# We copied the function from history_recorder.jl and injected max_steps into policy if required.

get_max_steps(p::POMCPOWPlanner) = p.solver.max_depth
get_max_steps(p::DPWPlanner) = p.solver.depth
get_max_steps(p::ActionGradMCTSPlanner) = p.solver.dpw_solver.depth

function set_max_steps!(p::POMCPOWPlanner, max_steps)
    p.solver.max_depth = max_steps
end
function set_max_steps!(p::DPWPlanner, max_steps)
    p.solver.depth = max_steps
end
function set_max_steps!(p::ActionGradMCTSPlanner, max_steps)
    p.solver.dpw_solver.depth = max_steps
end

function Base.iterate(it::MDPSimIterator{SPEC,M,P,RNG,S}, is::Tuple{Int,S}=(1, it.init_state)) where {SPEC,M<:MDP,P<:Union{DPWPlanner,ActionGradMCTSPlanner},RNG<:AbstractRNG,S}
    if isterminal(it.mdp, is[2]) || is[1] > it.max_steps
        return nothing
    end
    t = is[1]
    s = is[2]
    # We want the max_steps of the policy not to be greater than the remaining steps of the simulation.
    # We add 1 to the max_steps because the iterations run from 1 to N inclusive.
    curr_max_steps = get_max_steps(it.policy)
    set_max_steps!(it.policy, min(it.max_steps - t + 1, curr_max_steps))
    a, ai = action_info(it.policy, s)
    out = @gen(:sp, :r, :info)(it.mdp, s, a, it.rng)
    nt = merge(NamedTuple{(:sp, :r, :info)}(out), (t=t, s=s, a=a, action_info=ai))
    # Fixing the max_steps back to the original value
    # Just in case someone wants to later use that policy elsewhere
    set_max_steps!(it.policy, curr_max_steps)
    return (out_tuple(it, nt), (t + 1, nt.sp))
end

function check_tree_q_vals(it, is, ai)
    if ai[:tree] isa MCTS.DPWTree
        tree = ai[:tree]
    elseif ai[:tree] isa ActionGradientMCTS.ActionGradTree
        tree = ai[:tree].dpw_tree
    end

    for i in 1:length(tree.q)
        if tree.q[i] > 100.1 || tree.q[i] < -100.1
        end
    end
end

function Base.iterate(it::POMDPSimIterator{SPEC,M,P,U,RNG,B,S}, is::Tuple{Int,S,B}=(1, it.init_state, it.init_belief)) where {SPEC,M<:POMDP,P<:Union{POMCPOWPlanner,DPWPlanner,ActionGradMCTSPlanner},U<:Updater,RNG<:AbstractRNG,S,B}
    if isterminal(it.pomdp, is[2]) || is[1] > it.max_steps
        return nothing
    end
    t = is[1]
    s = is[2]
    b = is[3]
    if isterminal(it.pomdp, b)
        @warn "Reached terminal belief before terminal state/horizon reached"
        return nothing
    end
    # We want the max_steps of the policy not to be greater than the remaining steps of the simulation.
    # We add 1 to the max_steps because the iterations run from 1 to N inclusive.
    curr_max_steps = get_max_steps(it.policy)
    set_max_steps!(it.policy, min(it.max_steps - t + 1, curr_max_steps))
    if it.policy isa Union{DPWPlanner,ActionGradMCTSPlanner}
        if it.policy.mdp isa GenerativeBeliefMDP
            b_planner = initialize_belief(it.policy.mdp.updater, b)
        elseif it.policy.mdp isa GenerativeBeliefPropMDP
            b_planner = initialize_belief(it.policy.mdp.gmdp.updater, b)
        else
            b_planner = b
        end
    else
        b_planner = b
    end
    a, ai = action_info(it.policy, b_planner)
    out = @gen(:sp, :o, :r, :info)(it.pomdp, s, a, it.rng)
    outnt = NamedTuple{(:sp, :o, :r, :info)}(out)
    bp = b
    ui = NamedTuple()
    try
        bp, ui = update_info(it.updater, b, a, outnt.o)
    catch ex
        println(stderr, "Error in update_info: $ex")
        ui = (; ex,)
    end
    nt = merge(outnt, (t=t, b=b, s=s, a=a, action_info=ai, bp=bp, update_info=ui))
    set_max_steps!(it.policy, curr_max_steps)
    return (out_tuple(it, nt), (t + 1, nt.sp, nt.bp))
end
