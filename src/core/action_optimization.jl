function optimize_action(optim::GradAscentActionOptimizer, sanode::Int, a::A, grad_a_q) where {A}
    return a + optim.step_size * grad_a_q
end

function optimize_action(optim::FluxOptimizer{A}, sanode::Int, a::A, grad_a_q) where {A}
    return optimize_node(optim, sanode, a, grad_a_q)
end

"""
Compute new actions based on a gradient-based optimizer.
Input: a, q grad w.r.t. a (i.e. grad_action_q).
Output: a^{\\prime} i.e. the new action.
"""
function action_grad_optim(planner::ActionGradMCTSPlanner, snode::Int, sanode::Int, d::Int, grad_a_q)
    return project_action(planner.mdp, optimize_action(planner.action_optimizer, sanode, a_labels(planner.tree, sanode), grad_a_q))
end

"""
Update the action branches based on the new action a_prime.

In the POMDP case, sp will be populated with the propagated belief (bminus) rather than the posterior belief.
"""
function action_optim(planner::ActionGradMCTSPlanner, snode::Int, sanode::Int, d::Int)
    rule = planner.action_update_rule
    update_action_branch = true
    add_posterior_sample = false
    n_delta_action_optim = 0

    if rule isa NoActionUpdate
        update_action_branch = false
    end
    if rule isa ActionUpdateAllTreeAfterSimulate
        update_action_branch = true
    end
    if rule isa ActionUpdateMinVisits
        update_action_branch = update_action_branch && n_visits(planner.tree, sanode) >= rule.min_visits
    end
    if rule isa ActionUpdateEveryK
        update_action_branch = update_action_branch && n_visits(planner.tree, sanode) % rule.k == 0
    end
    if rule isa ActionUpdateMinChildren
        update_action_branch = update_action_branch && length(transitions(planner.tree, sanode)) >= rule.min_children
    end

    if update_action_branch
        for i in 1:planner.solver.action_optim_iters
            grad_a_q, grad_log_pts = grad_action_q(planner, snode, sanode, d)
            a_prime = action_grad_optim(planner, snode, sanode, d, grad_a_q)
            min_a_dist = rule isa ActionUpdateMinDist ? rule.min_a_dist : 0.0
            max_a_dist = rule isa ActionUpdateMaxDist ? rule.max_a_dist : Inf

            if norm(a_labels(planner.tree, sanode) - a_prime) >= min_a_dist
                add_sample, n_delta = update_action_branches!(planner, snode, sanode, d, a_prime, grad_log_pts, max_a_dist)
                add_posterior_sample = add_posterior_sample || add_sample
                n_delta_action_optim += n_delta
            end

        end
    end

    return add_posterior_sample, n_delta_action_optim
end
