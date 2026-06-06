module ActionGradientMCTS

using POMDPs
import POMDPs: solve, action
using POMDPTools
import POMDPTools: action_info
using MCTS
import MCTS: simulate,
    DPWTree,
    DPWStateNode,
    children,
    n_children,
    isroot,
    insert_action_node!,
    insert_state_node!,
    best_sanode,
    best_sanode_UCB,
    clear_tree!,
    convert_estimator
using ParticleFilters
import ParticleFilters: n_particles, particle, AbstractParticleBelief
using LinearAlgebra: norm
import Flux
using StaticArrays
using Random
using StatsBase
using StatsFuns
import Lazy: @forward
using ProgressMeter
import POMDPTools.ModelTools: UnderlyingMDP,
    GenerativeBeliefMDP,
    determine_gbmdp_state_type,
    BackwardCompatibleTerminalBehavior

include("core/interfaces.jl")
include("core/belief_mdp.jl")
include("core/optimizers.jl")
include("core/update_rules.jl")
include("core/value_updaters.jl")
include("core/solver.jl")
include("core/tree.jl")
include("core/planner.jl")
include("core/value_updates.jl")
include("core/action_updates.jl")
include("core/action_optimization.jl")
include("core/search.jl")

export ActionGradMCTSSolver,
    ActionGradMCTSPlanner,
    ActionGradTree,
    ActionGradStateNode,
    AbstractActionOptimizer,
    GradAscentActionOptimizer,
    FluxOptimizer,
    AbstractActionUpdateRule,
    NoActionUpdate,
    ActionUpdateAllTreeAfterSimulate,
    ActionUpdateMinVisitations,
    ActionUpdateMinEveryKVisits,
    ActionUpdateMinEveryKMinADist,
    ActionUpdateMinEveryKMaxADist,
    ActionUpdateMinChildrenEveryKMinADist,
    ActionUpdateMinChildrenEveryKMaxADist,
    AbstractValueUpdater,
    ValueUpdaterSNMISMC,
    GenerativeBeliefPropMDP,
    solve,
    action,
    action_info,
    clear_tree!,
    create_tree!,
    isroot,
    total_n_visits,
    children,
    n_children,
    s_labels,
    s_lookup,
    has_state_index,
    has_state,
    q_value,
    n_visits,
    transitions,
    a_labels,
    a_lookup,
    a_init,
    has_action,
    n_a_children,
    has_unique_transitions,
    sminus_labels,
    value,
    grad_log_transition,
    grad_transition_likelihood,
    grad_reward,
    transition_log_likelihood,
    project_action,
    logsumexp_factor,
    logsumexp_and_softmax

end
