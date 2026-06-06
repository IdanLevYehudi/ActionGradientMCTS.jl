abstract type AbstractActionOptimizer end
abstract type AbstractActionUpdateRule end
abstract type AbstractValueUpdater end

function update_value_visitation_count! end
function update_value_q! end
function update_action_branches! end
function optimize_action! end
function optimize_action end
function action_exploration_policy end
function grad_action_q end
function grad_log_transition end
function grad_transition_likelihood end
function grad_reward end
function project_action end
function transition_log_likelihood end

const BELIEF_MC_GRAD_LOG_TRANSITION = true
const LOG_P_UNINIT = 32.0
