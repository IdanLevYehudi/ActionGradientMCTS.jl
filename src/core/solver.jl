mutable struct ActionGradMCTSSolver{T<:AbstractActionOptimizer,U<:AbstractActionUpdateRule,V<:AbstractValueUpdater} <: AbstractMCTSSolver
    dpw_solver::DPWSolver
    action_optimizer::T
    action_optim_iters::Int
    grad_mc_k_s::Int
    grad_mc_k_obs::Int
    grad_mc_k_particles::Int
    use_mc_immediate_reward::Bool
    sample_state_reward_gradient::Bool
    linearize_weight_update::Bool
    grad_weighted_by_visits::Bool
    action_update_rule::U
    update_actions_imp_ratio_add_threshold::Float64
    update_actions_imp_ratio_delete_threshold::Float64
    optimize_before_update::Bool
    choose_random_obs::Bool
    value_updater::V
end

@forward_fields ActionGradMCTSSolver :dpw_solver (
    :depth,
    :exploration_constant,
    :n_iterations,
    :max_time,
    :k_action,
    :alpha_action,
    :k_state,
    :alpha_state,
    :keep_tree,
    :enable_action_pw,
    :enable_state_pw,
    :check_repeat_state,
    :check_repeat_action,
    :tree_in_info,
    :rng,
    :estimate_value,
    :init_Q,
    :init_N,
    :next_action,
    :default_action,
    :reset_callback,
    :show_progress,
    :timer)

function ActionGradMCTSSolver(; dpw_solver::DPWSolver=DPWSolver(),
    action_optimizer=GradAscentActionOptimizer(0.01),
    action_optim_iters=5,
    grad_mc_k_s=3,
    grad_mc_k_obs=2,
    grad_mc_k_particles=3,
    use_mc_immediate_reward=true,
    sample_state_reward_gradient=true,
    linearize_weight_update=true,
    grad_weighted_by_visits=false,
    action_update_rule=ActionUpdateAllTreeAfterSimulate(),
    update_actions_imp_ratio_add_threshold=0.75,
    update_actions_imp_ratio_delete_threshold=5e-2,
    optimize_before_update=true,
    choose_random_obs=false,
    value_updater=ValueUpdaterSNMISMC()
)
    return ActionGradMCTSSolver{typeof(action_optimizer),typeof(action_update_rule),typeof(value_updater)}(dpw_solver, action_optimizer, action_optim_iters, grad_mc_k_s, grad_mc_k_obs, grad_mc_k_particles, use_mc_immediate_reward, sample_state_reward_gradient, linearize_weight_update, grad_weighted_by_visits, action_update_rule, update_actions_imp_ratio_add_threshold, update_actions_imp_ratio_delete_threshold, optimize_before_update, choose_random_obs, value_updater)
end
