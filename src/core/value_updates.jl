function update_value_visitation_count!(up::ValueUpdaterSNMISMC, tree::ActionGradTree, snode::Int, sanode::Int, spnode::Int, r::Float64, gamma::Float64, n_delta_prime::Int=0)
    new_total_n_child = total_n_visits(tree, spnode)
    # Note: We assume that n_visits is initialized to 0 (and not 1) to a newly opened posterior node
    total_n_child = up.prev_total_n[spnode]
    n_delta = new_total_n_child - total_n_child
    current_n = n_visits(tree, sanode)
    new_n = current_n + n_delta
    total_n = total_n_visits(tree, snode)
    new_total_n = total_n + n_delta

    new_sn_log_denom, new_sn_log_denom_sign = logsumexp_factor([up.sn_log_denom[sanode], up.p_log_likelihoods[spnode] - up.q_log_likelihoods[spnode]], [1.0, Float64(n_delta)])

    # Update the immediate reward estimate
    prev_child_v = up.prev_value[spnode]
    new_child_v = up.value[spnode]
    new_log_future_v, new_sgn_future_v = logsumexp_factor([up.sn_log_denom[sanode], up.p_log_likelihoods[spnode] - up.q_log_likelihoods[spnode]], [up.future_v_est[sanode], new_child_v * (new_total_n_child + 1) - prev_child_v * (total_n_child + 1)])
    new_future_v_estimate = new_sgn_future_v * exp(new_log_future_v - new_sn_log_denom)

    new_log_r, new_sgn_r = logsumexp_factor([up.sn_log_denom[sanode], up.p_log_likelihoods[spnode] - up.q_log_likelihoods[spnode]], [up.r_est[sanode], r * n_delta])
    new_r = new_sgn_r * exp(new_log_r - new_sn_log_denom)

    # Update the future value estimates
    # Based on the equation:
    # \tilde{V}_{t+1}(b_{t},a_{t}^{\prime})=\frac{1}{N^{\prime}}(N\hat{V}_{t+1}(b_{t},a_{t})\\+(\boldsymbol{p}\oslash\boldsymbol{q})[j]\left((\boldsymbol{n}\odot\boldsymbol{v})^{\prime}[j]-(\boldsymbol{n}\odot\boldsymbol{v})[j]\right))
    new_q_estimate = new_r + gamma * new_future_v_estimate
    new_value = (total_n * up.value[snode] + new_n * new_q_estimate - current_n * q_value(tree, sanode)) / new_total_n

    # Updating visitation counts for the current action edge (sanode) and belief node (snode)
    tree.dpw_tree.n[sanode] = new_n
    tree.dpw_tree.total_n[snode] = new_total_n
    up.prev_total_n[snode] = total_n - n_delta_prime

    # Set the new future value estimate and q estimate
    up.r_est[sanode] = new_r
    up.future_v_est[sanode] = new_future_v_estimate
    tree.dpw_tree.q[sanode] = new_q_estimate
    up.sn_log_denom[sanode] = new_sn_log_denom

    # Update the future value, value and the q estimates
    # Value update for the weighted mean estimator:
    # \tilde{V}_{t}(b_{t})=\hat{V}_{t}(b_{t})-\frac{n(b_{t},a_{t})}{n(b_{t})}\hat{Q}(b_{t},a_{t})+\frac{n^{\prime}(b_{t},a_{t})}{n^{\prime}(b_{t})}\tilde{Q}(b_{t},a_{t}^{\prime})
    up.prev_value[snode] = up.value[snode]
    up.value[snode] = new_value
end

function mc_immediate_reward(planner::ActionGradMCTSPlanner{P,S,A,SE,NA,RCB,RNG}, s::S, a::A, sminus::S) where {P<:Union{MDP,POMDP},S,A,SE,NA,RCB,RNG}
    mdp = planner.mdp
    m, is_belief_mdp = mdp_or_pomdp_of_belief_mdp(mdp)
    if is_belief_mdp
        k = planner.solver.grad_mc_k_s
        particle_indices = rand(planner.rng, 1:n_particles(sminus), k)
        # Technically for the reward, this needs to be a random index from s, which could be a weighted particle filter, and then taking the corresponding particle from s and sminus.
        # Practically, s and sminus are unweighted particle filters in the current implementation, for which this is equivalent.
        new_r = 0.0
        for i in particle_indices
            if !isterminal(m, particle(s, i))
                new_r += reward(m, particle(s, i), a, particle(sminus, i))
            end
        end
        new_r /= k
    else  # Otherwise it is a regular MDP
        new_r = reward(mdp, s, a, sminus)
    end
    return new_r
end

is_generative_belief_prop_mdp(mdp) = mdp isa GenerativeBeliefPropMDP

function mdp_or_pomdp_of_belief_mdp(mdp::Union{MDP,POMDP})
    # Recover the underlying MDP/POMDP for both propagated and standard belief MDP wrappers.
    is_belief_mdp = false
    if is_generative_belief_prop_mdp(mdp)
        m = UnderlyingMDP(mdp.gmdp.pomdp)
        is_belief_mdp = true
    elseif mdp isa POMDPTools.ModelTools.GenerativeBeliefMDP
        m = UnderlyingMDP(mdp.pomdp)
        is_belief_mdp = true
    else
        m = mdp
    end
    return m, is_belief_mdp
end
