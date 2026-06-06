mutable struct ActionGradMCTSPlanner{P<:Union{MDP,POMDP},S,A,SE,NA,RCB,RNG} <: AbstractMCTSPlanner{P}
    solver::ActionGradMCTSSolver
    mdp::P
    tree::Union{Nothing,ActionGradTree{S,A}}
    solved_estimate::SE
    next_action::NA
    reset_callback::RCB
    rng::RNG
    action_optimizer::AbstractActionOptimizer
    action_update_rule::AbstractActionUpdateRule
end

function ActionGradMCTSPlanner(solver::ActionGradMCTSSolver, mdp::P) where {P<:Union{POMDP,MDP}}
    se = convert_estimator(solver.dpw_solver.estimate_value, solver, mdp)
    return ActionGradMCTSPlanner{P,
        statetype(P),
        actiontype(P),
        typeof(se),
        typeof(solver.next_action),
        typeof(solver.reset_callback),
        typeof(solver.rng)}(solver,
        mdp,
        nothing,
        se,
        solver.next_action,
        solver.reset_callback,
        solver.rng,
        deepcopy(solver.action_optimizer),
        deepcopy(solver.action_update_rule)
    )
end

Random.seed!(p::ActionGradMCTSPlanner, seed) = Random.seed!(p.rng, seed)

"""
Delete existing decision tree.
"""
function clear_tree!(p::ActionGradMCTSPlanner)
    p.tree = nothing
end

# """
# Implement for environments that require resetting between planning sessions
# """
# reset_environment(m::Union{MDP,POMDP}) = nothing

"""
Create a new empty decision tree.
"""
function create_tree!(p::ActionGradMCTSPlanner{P,S,A,SE,NA,RCB,RNG}) where {P,S,A,SE,NA,RCB,RNG}
    tree = ActionGradTree{S,A}(p.solver.n_iterations, typeof(p.solver.value_updater)(p.solver.n_iterations))
    # reset_environment(p.mdp)
    p.tree = tree
    p.action_optimizer = deepcopy(p.solver.action_optimizer)
    p.action_update_rule = deepcopy(p.solver.action_update_rule)
end

function update_value_visitation_count!(tree::ActionGradTree, snode::Int, sanode::Int, spnode::Int, r::Float64, gamma::Float64, n_delta::Int=0)
    update_value_visitation_count!(tree.value_updater, tree, snode, sanode, spnode, r, gamma, n_delta)
end

function update_action_branches!(planner::ActionGradMCTSPlanner{P,S,A,SE,NA,RCB,RNG}, snode::Int, sanode::Int, d::Int, a_prime::A, grad_log_pts::Vector{A}, max_a_dist::Float64=Inf) where {P<:Union{MDP,POMDP},S,A,SE,NA,RCB,RNG}
    # First update q based on the equation
    update_action_branches!(planner.tree.value_updater, planner, snode, sanode, d, a_prime, grad_log_pts, max_a_dist)
end

function update_new_posterior_child!(planner::ActionGradMCTSPlanner{P,S,A,SE,NA,RCB,RNG}, snode::Int, sanode::Int, spnode::Int, v::Float64) where {P<:Union{MDP,POMDP},S,A,SE,NA,RCB,RNG}
    update_new_posterior_child!(planner.tree.value_updater, planner, snode, sanode, spnode, v)
end

function update_terminal_node!(tree::ActionGradTree, snode::Int, v::Float64, n_delta::Int=1)
    update_terminal_node!(tree.value_updater, tree, snode, v, n_delta)
end

"""
Compute ``\\nabla_{a}Q_{t}^{\\pi}(x_{t},a)`` for a given state node and action pair.

Required calls:
 1. grad_log_transition(m::Union{MDP,POMDP}, s::S, a::A, sp::S)
 2. grad_reward(m::Union{MDP,POMDP}, s::S, a::A, sp::S)

In the POMDP case, the problem should return the propagated grad-log-transition likelihood for a received propagated belief given a starting belief and action (not the posterior likelihood).
"""
function grad_action_q(planner::ActionGradMCTSPlanner, snode::Int, sanode::Int, d::Int)
    grad_action_q(planner.tree.value_updater, planner, snode, sanode, d)
end
