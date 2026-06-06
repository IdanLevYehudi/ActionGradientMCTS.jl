function update_action_branches!(up::ValueUpdaterSNMISMC, planner::ActionGradMCTSPlanner{P,S,A,SE,NA,RCB,RNG}, snode::Int, sanode::Int, d::Int, a_prime::A, grad_log_pts::Vector{A}, max_a_dist::Float64=Inf) where {P<:Union{MDP,POMDP},S,A,SE,NA,RCB,RNG}
    # 1. Recompute the immediate reward estimate
    # 2. Recompute p_log_likelihoods
    # 3. Recompute future value estimates
    # 4. Recompute q values
    # 5. Recompute value estimates

    tree = planner.tree
    mdp = planner.mdp
    new_r_est = 0.0  # Will add the weighted mean of immediate reward
    new_future_v_est = 0.0  # Will add the weighted mean of future value
    new_sn_log_w_norm = 0.0

    m, is_belief_mdp = mdp_or_pomdp_of_belief_mdp(mdp)
    linearize_weight_update = planner.solver.linearize_weight_update

    ## If the distance of a_prime is greater than max_a_dist, normalize in the direction of the update
    current_a = a_labels(tree, sanode)
    if max_a_dist < Inf
        update_vec = a_prime - current_a
        norm_update = norm(update_vec)
        if norm_update > max_a_dist
            update_vec *= max_a_dist / norm_update
            a_prime = current_a + update_vec
        end
    end
    δa = a_prime - current_a

    add_posterior_sample = true

    s = s_labels(tree, snode)
    new_transitions = empty(transitions(tree, sanode))
    new_n = 0
    for i in 1:length(transitions(tree, sanode))
        spnode, old_r = transitions(tree, sanode)[i]
        sminus = sminus_labels(tree, spnode)
        sampled_s_sp = sampled_s_sps_labels(tree, spnode)  # The sampled state and posterior state tuple. For POMDPs these are actual state samples, and not beliefs.


        if linearize_weight_update
            q_i = up.q_log_likelihoods[spnode]
            if q_i == LOG_P_UNINIT  # If q_i hasn't been initialized, calculate it based on current action
                q_i = 0.0  # Since in this case we're only making relative weight updates, there is no need to compute the denominator
                up.q_log_likelihoods[spnode] = q_i
                p_i = 0.0
                up.p_log_likelihoods[spnode] = p_i
            end
            # Linearized update of importance ratio numerator
            # Since log(w) = log(p_i) - log(q_i)
            # log(w') = δlog(p_i) + log(w)
            # And the linearization is δlog(p_i) ≈ grad_log_p_i' * δa
            delta_log_pi = grad_log_pts[i]' * δa
            p_i = up.p_log_likelihoods[spnode] = up.p_log_likelihoods[spnode] + delta_log_pi
        else
            q_i = up.q_log_likelihoods[spnode]
            if q_i == LOG_P_UNINIT  # If q_i hasn't been initialized, calculate it based on current action
                a = a_labels(tree, sanode)
                q_i = transition_log_likelihood(mdp, s, a, sminus)
                up.q_log_likelihoods[spnode] = q_i
            end
            p_i = transition_log_likelihood(mdp, s, a_prime, sminus)
            up.p_log_likelihoods[spnode] = p_i
        end


        # Decide whether the updated branch still has enough importance mass.
        imp_ratio = exp(p_i - q_i)
        if imp_ratio >= planner.solver.update_actions_imp_ratio_add_threshold
            add_posterior_sample = false
        end
        if imp_ratio < planner.solver.update_actions_imp_ratio_delete_threshold

            continue
        end
        if planner.solver.use_mc_immediate_reward
            new_r = mc_immediate_reward(planner, s, a_prime, sminus)
        else
            new_r = reward(m, sampled_s_sp[1], a_prime, sampled_s_sp[2])
        end
        new_n += total_n_visits(tree, spnode) + 1
        # Update the immediate reward on the action edge
        push!(new_transitions, (spnode, new_r))
    end


    empty_transitions = false

    if isempty(new_transitions)
        empty_transitions = true
        maximal_spnode_ind = argmax([(up.p_log_likelihoods[spnode] - up.q_log_likelihoods[spnode]) for (spnode, r) in transitions(tree, sanode)])
        maximal_spnode = transitions(tree, sanode)[maximal_spnode_ind][1]
        sminus = sminus_labels(tree, maximal_spnode)
        sampled_s_sp = sampled_s_sps_labels(tree, maximal_spnode)  # The sampled state and posterior state tuple. For POMDPs these are actual state samples, and not beliefs.
        if planner.solver.use_mc_immediate_reward
            new_r = mc_immediate_reward(planner, s, a_prime, sminus)
        else
            new_r = reward(m, sampled_s_sp[1], a_prime, sampled_s_sp[2])
        end
        push!(new_transitions, (maximal_spnode, new_r))
        new_n = total_n_visits(tree, maximal_spnode) + 1
    end

    new_n_a_children = length(new_transitions)  # The number of unique observation branches we have up until now

    arr_new_sn_log_w_norm = [(up.p_log_likelihoods[spnode] - up.q_log_likelihoods[spnode]) for (spnode, r) in new_transitions]
    new_sn_log_w_norm, new_w_sampling = logsumexp_and_softmax(arr_new_sn_log_w_norm)
    log_n_visits = [log(total_n_visits(tree, spnode) + 1) for (spnode, r) in new_transitions]

    arr_new_sn_log_denom = log_n_visits .+ arr_new_sn_log_w_norm
    new_sn_log_denom = logsumexp(arr_new_sn_log_denom)

    new_log_r, new_sgn_r = logsumexp_factor(arr_new_sn_log_w_norm, [((total_n_visits(tree, spnode) + 1) * r) for (spnode, r) in new_transitions])
    new_log_future_v, new_sgn_future_v = logsumexp_factor(arr_new_sn_log_w_norm, [((total_n_visits(tree, spnode) + 1) * value(tree, spnode)) for (spnode, r) in new_transitions])

    new_r_est = new_sgn_r * exp(new_log_r - new_sn_log_denom)
    new_future_v_est = new_sgn_future_v * exp(new_log_future_v - new_sn_log_denom)

    current_q_estimate = q_value(tree, sanode)
    new_q_estimate = new_r_est + discount(mdp) * new_future_v_est

    current_n = n_visits(tree, sanode)
    n_delta = new_n - current_n
    total_n = total_n_visits(tree, snode)
    new_total_n = total_n + n_delta

    new_value = (total_n * up.value[snode] + new_n * new_q_estimate - current_n * current_q_estimate) / new_total_n

    # Update the action label to a_prime
    set_a_label!(tree, snode, sanode, a_prime)

    push!(a_init(tree, sanode), a_prime)

    for i in 1:length(new_transitions)
        spnode, r = new_transitions[i]
        up.w_sampling[spnode] = new_w_sampling[i]
    end

    if planner.solver.dpw_solver.check_repeat_state
        new_unique_transitions = unique((sanode, spnode) for (spnode, r) in new_transitions)
        tree.dpw_tree.unique_transitions = new_unique_transitions
    end

    # Set the new q and value estimates
    up.r_est[sanode] = new_r_est
    up.future_v_est[sanode] = new_future_v_est
    up.sn_log_denom[sanode] = new_sn_log_denom
    up.sn_log_w_norm[sanode] = new_sn_log_w_norm

    # Updating visitation counts for the current action edge (sanode) and belief node (snode)
    tree.dpw_tree.n[sanode] = new_n
    tree.dpw_tree.q[sanode] = new_q_estimate
    tree.dpw_tree.transitions[sanode] = new_transitions
    tree.dpw_tree.n_a_children[sanode] = new_n_a_children

    up.prev_total_n[snode] = tree.dpw_tree.total_n[snode]
    tree.dpw_tree.total_n[snode] = new_total_n
    up.prev_value[snode] = up.value[snode]
    up.value[snode] = new_value

    return add_posterior_sample, n_delta
end

function set_prev_value_updater!(planner::ActionGradMCTSPlanner{P,S,A,SE,NA,RCB,RNG}, snode::Int, prev_total_n::Int, prev_value::Float64) where {P<:Union{MDP,POMDP},S,A,SE,NA,RCB,RNG}
    set_prev_value_updater!(planner.tree.value_updater, snode, prev_total_n, prev_value)
end

function set_prev_value_updater!(up::ValueUpdaterSNMISMC, snode::Int, prev_total_n::Int, prev_value::Float64)
    up.prev_total_n[snode] = prev_total_n
    up.prev_value[snode] = prev_value
end

function update_new_posterior_child!(up::ValueUpdaterSNMISMC, planner::ActionGradMCTSPlanner{P,S,A,SE,NA,RCB,RNG}, snode::Int, sanode::Int, spnode::Int, v::Float64) where {P<:Union{MDP,POMDP},S,A,SE,NA,RCB,RNG}
    up.value[spnode] = v
        up.q_log_likelihoods[spnode] = up.p_log_likelihoods[spnode] = LOG_P_UNINIT

    # Need to update w_sampling for all siblings, based on new sn_log_w_norm
    new_sn_log_w_norm = logaddexp(up.sn_log_w_norm[sanode], up.p_log_likelihoods[spnode] - up.q_log_likelihoods[spnode])
    up.w_sampling[spnode] = exp(up.p_log_likelihoods[spnode] - up.q_log_likelihoods[spnode] - new_sn_log_w_norm)
    for (sibling_spnode, r) in transitions(planner.tree, sanode)
        if sibling_spnode == spnode
            continue
        end
        up.w_sampling[sibling_spnode] = exp(log(up.w_sampling[sibling_spnode]) + up.sn_log_w_norm[sanode] - new_sn_log_w_norm)
    end
    up.sn_log_w_norm[sanode] = new_sn_log_w_norm
end

function update_terminal_node!(up::ValueUpdaterSNMISMC, tree::ActionGradTree, snode::Int, v::Float64, n_delta::Int=1)
    total_n = total_n_visits(tree, snode)
    new_total_n = total_n + n_delta

    # The denominator is +1 because for a posterior node, we need to remember that the visitaion count is always 1 less than the actual count.
    # In a non-terminal posterior node, this is desired in order to "forget" the initial value estimate of the rollout when actually visiting the node.
    # In a terminal node, the rollout value is all we have, so we simply want to average those values (including the first value).
    new_value = up.value[snode] + n_delta * (v - up.value[snode]) / (new_total_n + 1)

    up.value[snode] = new_value
    tree.dpw_tree.total_n[snode] = new_total_n
    up.prev_total_n[snode] = total_n
end

function grad_action_q(up::ValueUpdaterSNMISMC, planner::ActionGradMCTSPlanner, snode::Int, sanode::Int, d::Int)
    tree = planner.tree

    # Check if the sanode has been visited and has an actual q-value estimate
    if n_visits(tree, sanode) == 0
        return 0.0
    end

    mdp = planner.mdp
    s = s_labels(tree, snode)  # Current state node label.
    a = a_labels(tree, sanode)  # Action label for transition.

    grad_a_q = zero(a)
    grad_rs = Vector{typeof(a)}()  # The reward gradients
    grad_vs = Vector{typeof(a)}()  # The value gradients
    log_imp_ratios = Vector{Float64}()  # The importance ratios
    total_n_visits_arr = Vector{Float64}()  # The number of visits to the posterior state node
    grad_log_pts = Vector{typeof(a)}()  # We might preallocate but usually the number of branches is quite small so I'm not sure it's that worth it

    # Checking if reward gradient should be calculated based on new random samples
    grad_r = zero(a)
    grad_v = zero(a)
    m, is_belief_mdp = mdp_or_pomdp_of_belief_mdp(mdp)

    # baseline_subtraction = 0.0  # Using regular Q-gradient
    baseline_subtraction = up.value[snode]  # Using advantage gradient

    grad_mc_k_s = planner.solver.grad_mc_k_s
    linearize_weight_update = planner.solver.linearize_weight_update
    sample_state_reward_gradient = planner.solver.sample_state_reward_gradient
    if sample_state_reward_gradient
        for i in 1:grad_mc_k_s
            if is_belief_mdp
                s_start = rand(s)
            else
                s_start = s
            end
            sp, r = @gen(:sp, :r)(m, s_start, a, planner.rng)
            grad_r += grad_log_transition(m, s_start, a, sp) * r + grad_reward(m, s_start, a, sp)
        end
        grad_r /= grad_mc_k_s
    end

    if linearize_weight_update
        # In this scheme, we update the log importance weights based on the gradients we calculate here.
        # Therefore, we need to go through all of the branches and compute grad_log_transition (for caching).
        for (spnode, r) in transitions(tree, sanode)
            sminus = sminus_labels(tree, spnode)  # The posterior state node label - might be propagated belief in the POMDP case.
            sampled_s_sp = sampled_s_sps_labels(tree, spnode)  # The sampled state and posterior state tuple. For POMDPs these are actual state samples, and not beliefs.
                        if BELIEF_MC_GRAD_LOG_TRANSITION && is_belief_mdp
                # Note that we take a random index without taking into account the weights.
                # This is because the original computation is the sum over all particles (without being weighted).
                # We reweigh by n_particles to ensure that in expectation we get the same sum.
                global BELIEF_MC_GRAD_LOG_TRANSITION_RAND_PARTICLE
                if BELIEF_MC_GRAD_LOG_TRANSITION_RAND_PARTICLE && is_belief_mdp
                    k = planner.solver.grad_mc_k_particles
                    particle_indices = rand(planner.rng, 1:n_particles(sminus), k)
                    grad_log_pt = n_particles(sminus) * sum(grad_log_transition(m, particle(s, i), a, particle(sminus, i)) for i in particle_indices) / k
                    if !sample_state_reward_gradient
                        curr_grad_r = sum(grad_reward(m, particle(s, i), a, particle(sminus, i)) for i in particle_indices) / k
                    end
                else
                    grad_log_pt = n_particles(sminus) * grad_log_transition(m, sampled_s_sp[1], a, sampled_s_sp[2])
                    curr_grad_r = grad_reward(m, sampled_s_sp[1], a, sampled_s_sp[2])
                end
            else
                grad_log_pt = grad_log_transition(mdp, s, a, sminus)
                curr_grad_r = grad_reward(mdp, sampled_s_sp[1], a, sampled_s_sp[2])
            end

            if !all(isa.(grad_log_pt, Number)) || !all(isfinite.(grad_log_pt))
                @warn "grad_log_pt not a finite vector" grad_log_pt s a sminus
                continue
            end
            v = value(tree, spnode)

            curr_grad_v = grad_log_pt .* ((r + discount(mdp) * v) - baseline_subtraction)  # Generalized baseline subtraction for variance reduction

            log_imp_ratio = up.p_log_likelihoods[spnode] - up.q_log_likelihoods[spnode]
            curr_total_n_visits = total_n_visits(tree, spnode) + 1
            push!(log_imp_ratios, log_imp_ratio)
            push!(total_n_visits_arr, curr_total_n_visits)
            push!(grad_vs, curr_grad_v)
            push!(grad_log_pts, grad_log_pt)
            if !sample_state_reward_gradient
                push!(grad_rs, curr_grad_r)
            end
        end
        if !sample_state_reward_gradient
            grad_r = mean(grad_rs)
        end
        grad_v = mean(grad_vs)  # For debugging purposes
        if planner.solver.grad_weighted_by_visits
            ## Weighting gradient by (visitation_count) * (importance_ratio)
            if !sample_state_reward_gradient
                grad_a_q_log, grad_a_q_sign = logsumexp_factor(log_imp_ratios, total_n_visits_arr .* (grad_vs .+ grad_rs))
                grad_a_q = grad_a_q_sign .* exp.(grad_a_q_log .- up.sn_log_denom[sanode])
            else
                grad_a_q_log, grad_a_q_sign = logsumexp_factor(log_imp_ratios, total_n_visits_arr .* grad_vs)
                grad_a_q = grad_r + grad_a_q_sign .* exp.(grad_a_q_log .- up.sn_log_denom[sanode])
            end
        else
            ## Weighting gradient by importance_ratio only
            if !sample_state_reward_gradient
                weights_spnodes = [up.w_sampling[spnode] for (spnode, r) in transitions(tree, sanode)]
                grad_a_q = sum(weights_spnodes .* (grad_vs .+ grad_rs))
            else
                weights_spnodes = [up.w_sampling[spnode] for (spnode, r) in transitions(tree, sanode)]
                grad_a_q = grad_r + sum(weights_spnodes .* grad_vs)
            end
        end
    else
        # In this scheme, we will perform exact weight updates after the action update. Therefore we don't have to compute grad_log_pts for all branches.
        iteration_spnodes = transitions(tree, sanode)
        weights_spnodes = [up.w_sampling[spnode] for (spnode, r) in iteration_spnodes]

        # Instead of weighting the samples by the weights and sampling uniformly, we sample by the weights. This means fewer samples are needed (on average) in order to choose high-weight samples.
        iteration_spnodes = sample(planner.rng, iteration_spnodes, Weights(weights_spnodes, 1.0), planner.solver.grad_mc_k_obs)
        for (spnode, r) in iteration_spnodes
            sminus = sminus_labels(tree, spnode)  # The posterior state node label - might be propagated belief in the POMDP case.
            sampled_s_sp = sampled_s_sps_labels(tree, spnode)  # The sampled state and posterior state tuple. For POMDPs these are actual state samples, and not beliefs.
                        if BELIEF_MC_GRAD_LOG_TRANSITION && is_belief_mdp
                # Note that we take a random index without taking into account the weights.
                # This is because the original computation is the sum over all particles (without being weighted).
                # We reweigh by n_particles to ensure that in expectation we get the same sum.
                global BELIEF_MC_GRAD_LOG_TRANSITION_RAND_PARTICLE
                if BELIEF_MC_GRAD_LOG_TRANSITION_RAND_PARTICLE
                    k = planner.solver.grad_mc_k_particles
                    particle_indices = rand(planner.rng, 1:n_particles(sminus), k)
                    grad_log_pt = n_particles(sminus) * sum(grad_log_transition(m, particle(s, i), a, particle(sminus, i)) for i in particle_indices) / k
                    if !sample_state_reward_gradient
                        grad_r = sum(grad_reward(mdp, particle(s, i), a, particle(sminus, i)) for i in particle_indices) / k
                    end
                else
                    grad_log_pt = n_particles(sminus) * grad_log_transition(m, sampled_s_sp[1], a, sampled_s_sp[2])
                    if !sample_state_reward_gradient
                        grad_r = grad_reward(mdp, sampled_s_sp[1], a, sampled_s_sp[2])
                    end
                end
            else
                grad_log_pt = grad_log_transition(mdp, s, a, sminus)
                if !sample_state_reward_gradient
                    grad_r = grad_reward(mdp, sampled_s_sp[1], a, sampled_s_sp[2])
                end
            end

            if !all(isa.(grad_log_pt, Number)) || !all(isfinite.(grad_log_pt))
                @warn "grad_log_pt not a finite vector" grad_log_pt s a sminus
                continue
            end
            if !sample_state_reward_gradient
                v = value(tree, spnode)
                curr_grad = (grad_log_pt * (r + discount(mdp) * v - baseline_subtraction) + grad_r)
                grad_a_q += curr_grad
            else
                grad_v += grad_log_pt * (discount(mdp) * value(tree, spnode) - baseline_subtraction)
            end
        end

        if !sample_state_reward_gradient
            grad_a_q = grad_a_q ./ planner.solver.grad_mc_k_obs
        else
            grad_v = grad_v / planner.solver.grad_mc_k_obs
            grad_a_q = grad_v + grad_r
        end
    end

    return grad_a_q, grad_log_pts

end
