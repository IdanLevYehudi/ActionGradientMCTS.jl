POMDPs.solve(solver::ActionGradMCTSSolver, mdp::Union{POMDP,MDP}) = ActionGradMCTSPlanner(solver, mdp)

"""
Construct an ActionGradTree and choose the best action.
"""
POMDPs.action(p::ActionGradMCTSPlanner, s) = first(action_info(p, s))

"""
Construct an MCTSDPW tree and choose the best action. Also output some information.
"""
function POMDPTools.action_info(p::ActionGradMCTSPlanner, s; tree_in_info=false)
    local a::actiontype(p.mdp)
    info = Dict{Symbol,Any}()
    try
        if isterminal(p.mdp, s)
            ex = """
                  MCTS cannot handle terminal states. action was called with
                  s = $s
                  """
            info[:exception] = ex
            info[:tree_queries] = 0
            info[:search_time] = 0
            info[:search_time_us] = 0
            info[:best_Q] = 0
            a = convert(actiontype(p.mdp), default_action(p.solver.default_action, p.mdp, s, ex))
            return a, info
        end

        if p.solver.keep_tree && p.tree !== nothing
            tree = p.tree
            if has_state(tree, s)
                snode = s_lookup(tree, s)
            else
                snode = insert_state_node!(tree, s, true)
            end
        else
            create_tree!(p)
            tree = p.tree
            snode = insert_state_node!(tree, s, p.solver.check_repeat_state)
        end

        timer = p.solver.timer
        p.solver.show_progress ? progress = Progress(p.solver.n_iterations) : nothing
        nquery = 0
        start_s = timer()
        for i = 1:p.solver.n_iterations
            nquery += 1
            simulate(p, snode, p.solver.depth) # (not 100% sure we need to make a copy of the state here)
            p.solver.show_progress ? next!(progress) : nothing
            if timer() - start_s >= p.solver.max_time
                p.solver.show_progress ? finish!(progress) : nothing
                break
            end
        end
        p.reset_callback(p.mdp, s) # Optional: leave the MDP in the current state.
        info[:search_time] = timer() - start_s
        info[:search_time_us] = info[:search_time] * 1e6
        info[:tree_queries] = nquery
        if p.solver.tree_in_info || tree_in_info
            info[:tree] = tree
        end

        sanode = best_sanode(tree, snode)
        a = a_labels(tree, sanode)  # choose action with highest approximate value
        info[:best_Q] = q_value(tree, sanode)  # export the approximate value for the action

    catch ex
        if ex isa InterruptException
            throw(ex)
        end
        a = convert(actiontype(p.mdp), default_action(p.solver.default_action, p.mdp, s, ex))
        info[:exception] = ex
        info[:tree_queries] = 0
        info[:search_time] = 0
        info[:search_time_us] = 0
        info[:best_Q] = 0
    end


    return a, info
end

"""
Return the reward for one iteration of ActionGradMCTS.
"""
function simulate(planner::ActionGradMCTSPlanner, snode::Int, d::Int)
    sol = planner.solver
    tree = planner.tree
    s = s_labels(tree, snode)
    planner.reset_callback(planner.mdp, s) # Optional: used to reset/reinitialize MDP to a given state.

    ter = isterminal(planner.mdp, s)

    if ter || d == 0
        if ter
            ret = 0.0
        elseif d == 0
            ret = estimate_value(planner.solved_estimate, planner.mdp, s, d)
        end

        update_terminal_node!(tree, snode, ret)
        return ret
    end

    # action progressive widening
    if planner.solver.enable_action_pw
        if length(children(tree, snode)) <= sol.k_action * total_n_visits(tree, snode)^sol.alpha_action # criterion for new action generation
            a = next_action(planner.next_action, planner.mdp, s, ActionGradStateNode(tree, snode)) # action generation step
            if !sol.check_repeat_action || !has_action(tree, snode, a)
                n0 = init_N(sol.init_N, planner.mdp, s, a)
                insert_action_node!(tree, snode, a, n0,
                    init_Q(sol.init_Q, planner.mdp, s, a),
                    sol.check_repeat_action
                )
                add_total_n_visits!(tree, snode, n0)
            end
        end
    elseif isempty(children(tree, snode))
        for a in actions(planner.mdp, s)
            n0 = init_N(sol.init_N, planner.mdp, s, a)
            insert_action_node!(tree, snode, a, n0,
                init_Q(sol.init_Q, planner.mdp, s, a),
                false)
            add_total_n_visits!(tree, snode, n0)
        end
    end

    sanode = best_sanode_UCB(tree, snode, sol.exploration_constant)
    a = a_labels(tree, sanode)
    # state progressive widening
    new_node = false

    snode_value_before = value(tree, snode)
    total_n_before = total_n_visits(tree, snode)

    n_delta_action_optim = 0
    state_pw_condition = (planner.solver.enable_state_pw && n_a_children(tree, sanode) <= sol.k_state * n_visits(tree, sanode)^sol.alpha_state) || n_a_children(tree, sanode) == 0
    if planner.solver.optimize_before_update
        add_posterior_sample, n_delta_action_optim = action_optim(planner, snode, sanode, d)
        state_pw_condition = state_pw_condition || add_posterior_sample
    end
    if state_pw_condition
        sp, r, info = @gen(:sp, :r, :info)(planner.mdp, s, a, planner.rng)
        if info === nothing || !haskey(info, :sminus)
            sminus = sp
        else
            sminus = info[:sminus]
        end
        if planner.solver.use_mc_immediate_reward
            r = mc_immediate_reward(planner, s, a, sminus)
        end
        if info === nothing || !haskey(info, :sampled_s_sp)
            sampled_s_sp = (s, sp)
        else
            sampled_s_sp = info[:sampled_s_sp]
        end

        if sol.check_repeat_state && has_state(tree, sp)
            spnode = s_lookup(tree, sp)
        else
            spnode = insert_state_node!(tree, sp, sol.keep_tree || sol.check_repeat_state, sminus, sampled_s_sp, 0.0, 0.0, a)
            new_node = true
        end

        add_transition!(tree, sanode, spnode, r)

        if !sol.check_repeat_state
            add_n_a_children!(tree, sanode, 1)
        elseif !(has_unique_transitions(tree, sanode, spnode))
            add_unique_transition!(tree, sanode, spnode)
            add_n_a_children!(tree, sanode, 1)
        end
    else
        if planner.solver.choose_random_obs
            # Choosing at random
            spnode, r = rand(planner.rng, transitions(tree, sanode))
        else
            # Choose child with least visits
            least_visited_sp = argmin([total_n_visits(tree, sp) for (sp, r) in transitions(tree, sanode)])
            spnode, r = transitions(tree, sanode)[least_visited_sp]
        end
    end

    if new_node
        v = estimate_value(planner.solved_estimate, planner.mdp, sp, d - 1)
        update_new_posterior_child!(planner, snode, sanode, spnode, v)
    else
        v = simulate(planner, spnode, d - 1)
    end
    gamma = discount(planner.mdp)

    q = r + gamma * v
    update_value_visitation_count!(tree, snode, sanode, spnode, r, gamma, n_delta_action_optim)
    if !planner.solver.optimize_before_update
        add_posterior_sample, n_delta_action_optim = action_optim(planner, snode, sanode, d)
    end

    set_prev_value_updater!(planner, snode, total_n_before, snode_value_before)

    return q
end
