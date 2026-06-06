
@with_kw mutable struct POMCPOWParams
    param_id::Int64 = 1
    max_query::Int64 = 1000
    max_tree_depth::Int64 = 10
    solver_timeout::Float64 = 1.0
    ucb_c::Float64 = 110.0
    k_act::Float64 = 3.0
    alpha_act::Float64 = 0.2
    k_obs::Float64 = 3.0
    alpha_obs::Float64 = 0.2
end

@with_kw mutable struct PFTDPWParams
    param_id::Int64 = 1
    max_query::Int64 = 1000
    max_tree_depth::Int64 = 10
    solver_timeout::Float64 = 1.0
    ucb_c::Float64 = 110.0
    k_act::Float64 = 5.0
    alpha_act::Float64 = 0.03
    k_obs::Float64 = 5.0
    alpha_obs::Float64 = 0.01
    num_particles_planner::Int64 = 50
    tree_in_info::Bool = false
end

@with_kw mutable struct ActionGradMCTSParams
    param_id::Int64 = 1
    max_query::Int64 = 1000
    max_tree_depth::Int64 = 10
    solver_timeout::Float64 = 1.0
    ucb_c::Float64 = 110.0
    k_act::Float64 = 5.0
    alpha_act::Float64 = 0.03
    k_obs::Float64 = 5.0
    alpha_obs::Float64 = 0.01
    num_particles_planner::Int64 = 50
    tree_in_info::Bool = false
    action_optimizer::AbstractActionOptimizer = GradAscentActionOptimizer(0.01)
    action_optim_iters::Int = 1
    grad_mc_k_s::Int = 1
    grad_mc_k_obs::Int = 1
    grad_mc_k_particles::Int = 1
    use_mc_immediate_reward::Bool = true
    sample_state_reward_gradient::Bool = true
    linearize_weight_update::Bool = true
    grad_weighted_by_visits::Bool = true
    action_update_rule::AbstractActionUpdateRule = ActionUpdateMinVisitations(1)
    update_actions_imp_ratio_add_threshold::Float64 = 0.75
    update_actions_imp_ratio_delete_threshold::Float64 = 5e-2
    optimize_before_update::Bool = true
    choose_random_obs::Bool = true
    value_updater::AbstractValueUpdater = ValueUpdaterSNMISMC()
end

abstract type ActionSampler end

struct RandomActionSampler <: ActionSampler
    rng::AbstractRNG
end

struct VOOActionSampler{V<:VOOActionGenerator} <: ActionSampler
    voo::V
end

@with_kw mutable struct SolverGenParams
    param_id::Int64 = 1
    solver_gen::Function
    num_particles_planner::Int64 = 50
    solver_name::String = "Unknown"
end

function solver_name_by_params(params, na; mdp::Bool=false)
    vpw = na isa VOOActionGenerator || (na isa PolicyFirstGen && na.gen isa VOOActionGenerator) || (na isa IteratorFirstGen && na.gen isa VOOActionGenerator)
    if params isa POMCPOWParams
        return canonical_solver_name(vpw ? "vomcpow" : "pomcpow"; mdp=mdp)
    elseif params isa PFTDPWParams
        return canonical_solver_name(vpw ? "vpw" : "dpw"; mdp=mdp)
    elseif params isa ActionGradMCTSParams
        return canonical_solver_name(vpw ? "ag-vpw" : "ag-dpw"; mdp=mdp)
    else
        return "Unknown"
    end
end

@with_kw mutable struct SimParams
    sim_max_steps::Int64 = 50
    num_sims::Int64 = 100
    num_particles_sim::Int64 = 1000
end

"""
Generate a set of POMDP simulations with POMCPOW solver, intialized with the given parameters.
# Arguments:
- `solver_params::POMCPOWParams` - parameters for solver
- `sim_params::SimParams` - parameters for simulation
- `estimate_value` - callable (i) -> estimate_value, for the ith simulation
- `next_action` - callable (i) -> next_action, for the ith simulation
- `default_action` - callable (i) -> default_action, for the ith simulation
"""
function gen_sims_pomcpow(; m::POMDP, solver_params::POMCPOWParams, sim_params::SimParams, estimate_value, next_action, default_action, seed=1234)
    sims = []
    for i in 1:sim_params.num_sims
        # Gather statistics over mean and standard deviation of the value (sum of discounted rewards)
        # For K times, run the simulation for N time steps
        rng_solver = Xoshiro(seed + 1237 + 30_000 * i)
        na = next_action(seed + 10_000 * i)
        solver = POMCPOWSolver(
            criterion=MaxUCB(solver_params.ucb_c),
            tree_queries=solver_params.max_query,
            max_depth=solver_params.max_tree_depth,
            max_time=solver_params.solver_timeout,
            k_action=solver_params.k_act,
            alpha_action=solver_params.alpha_act,
            k_observation=solver_params.k_obs,
            alpha_observation=solver_params.alpha_obs,
            estimate_value=estimate_value(seed + 10_000 * i),
            next_action=na,
            default_action=default_action(seed + 10_000 * i),
            rng=rng_solver,
            check_repeat_obs=false
        )
        planner = solve(solver, m)
        solver_name = solver_name_by_params(solver_params, na)

        filter = BootstrapFilter(
            deepcopy(m),
            sim_params.num_particles_sim,
            Xoshiro(seed + i * 90_000)
        )

        sim = Sim(
            deepcopy(m),
            planner,
            filter,
            rng=Xoshiro(seed + 50_000 * i),
            max_steps=sim_params.sim_max_steps,
            metadata=Dict(:solver => solver_name, :i => i, :k => solver_params.param_id)
        )
        push!(sims, sim)
    end

    return sims
end


"""
Generate a set of POMDP simulations with MCTS with DPW solver based on particle beliefs (AKA PFT-DPW), intialized with the given parameters.
# Arguments:
- `solver_params::PFTDPWParams` - parameters for solver
- `sim_params::SimParams` - parameters for simulation
- `estimate_value` - callable (i) -> estimate_value, for the ith simulation
- `next_action` - callable (i) -> next_action, for the ith simulation
- `default_action` - callable (i) -> default_action, for the ith simulation
"""
function gen_sims_pftdpw(; m::Union{MDP,POMDP}, solver_params::PFTDPWParams, sim_params::SimParams, estimate_value, next_action, default_action, seed=1234)
    sims = []
    for i in 1:sim_params.num_sims
        # Gather statistics over mean and standard deviation of the value (sum of discounted rewards)
        # For K times, run the simulation for N time steps
        if m isa POMDP
            rng_node_updater = Xoshiro(seed + 1234 + 30_000 * i)
            node_updater = BootstrapFilter(
                deepcopy(m),
                solver_params.num_particles_planner,
                rng_node_updater
            )
        end
        rng_solver = Xoshiro(seed + 1237 + 30_000 * i)
        na = next_action(seed + 10_000 * i)
        solver = DPWSolver(
            n_iterations=solver_params.max_query,
            exploration_constant=solver_params.ucb_c,
            depth=solver_params.max_tree_depth,
            max_time=solver_params.solver_timeout,
            k_action=solver_params.k_act,
            alpha_action=solver_params.alpha_act,
            k_state=solver_params.k_obs,
            estimate_value=estimate_value(seed + 10_000 * i),
            next_action=na,
            default_action=default_action(seed + 10_000 * i),
            rng=rng_solver,
            check_repeat_state=false,
            tree_in_info=solver_params.tree_in_info
        )
        solver_name = solver_name_by_params(solver_params, na; mdp=m isa MDP)
        if m isa MDP
            planner = solve(solver, m)
            sim = Sim(
                deepcopy(m),
                planner,
                rng=Xoshiro(seed + 50_000 * i),
                max_steps=sim_params.sim_max_steps,
                metadata=Dict(:solver => solver_name, :i => i, :k => solver_params.param_id)
            )
        else
            belief_mdp = GenerativeBeliefMDP(deepcopy(m), node_updater)
            planner = solve(solver, belief_mdp)
            rng_filter = Xoshiro(seed + i * 90_000)
            filter = BootstrapFilter(
                deepcopy(m),
                sim_params.num_particles_sim,
                rng_filter
            )
            sim = Sim(
                deepcopy(m),
                planner,
                filter,
                rng=Xoshiro(seed + 50_000 * i),
                max_steps=sim_params.sim_max_steps,
                metadata=Dict(:solver => solver_name, :i => i, :k => solver_params.param_id)
            )
        end
        push!(sims, sim)
    end
    return sims
end

"""
Generate a set of POMDP simulations with ActionGradMCTSSolver.
# Arguments:
- `solver_params::ActionGradMCTSParams` - parameters for solver
- `sim_params::SimParams` - parameters for simulation
- `estimate_value` - callable (i) -> estimate_value, for the ith simulation
- `next_action` - callable (i) -> next_action, for the ith simulation
- `default_action` - callable (i) -> default_action, for the ith simulation
"""
function gen_sims_actiongradmcts(; m::Union{MDP,POMDP}, solver_params::ActionGradMCTSParams, sim_params::SimParams, estimate_value, next_action, default_action, seed=1234)
    sims = []
    for i in 1:sim_params.num_sims
        m_copy = deepcopy(m)
        # Gather statistics over mean and standard deviation of the value (sum of discounted rewards)
        # For K times, run the simulation for N time steps
        if m isa POMDP
            rng_node_updater = Xoshiro(seed + 1234 + 30_000 * i)
            node_updater = BootstrapFilter(
                m_copy,
                solver_params.num_particles_planner,
                rng_node_updater
            )
        end
        rng_solver = Xoshiro(seed + 1237 + 30_000 * i)
        na = next_action(seed + 10_000 * i)
        solver = ActionGradMCTSSolver(dpw_solver=DPWSolver(
                n_iterations=solver_params.max_query,
                exploration_constant=solver_params.ucb_c,
                depth=solver_params.max_tree_depth,
                max_time=solver_params.solver_timeout,
                k_action=solver_params.k_act,
                alpha_action=solver_params.alpha_act,
                k_state=solver_params.k_obs,
                estimate_value=estimate_value(seed + 10_000 * i),
                next_action=na,
                default_action=default_action(seed + 10_000 * i),
                rng=rng_solver,
                check_repeat_state=false,
                tree_in_info=solver_params.tree_in_info),
            action_optimizer=solver_params.action_optimizer,
            action_optim_iters=solver_params.action_optim_iters,
            grad_mc_k_s=solver_params.grad_mc_k_s,
            grad_mc_k_obs=solver_params.grad_mc_k_obs,
            grad_mc_k_particles=solver_params.grad_mc_k_particles,
            grad_weighted_by_visits=solver_params.grad_weighted_by_visits,
            action_update_rule=solver_params.action_update_rule,
            update_actions_imp_ratio_add_threshold=solver_params.update_actions_imp_ratio_add_threshold,
            update_actions_imp_ratio_delete_threshold=solver_params.update_actions_imp_ratio_delete_threshold,
            choose_random_obs=solver_params.choose_random_obs,
            value_updater=solver_params.value_updater
        )
        solver_name = solver_name_by_params(solver_params, na; mdp=m isa MDP)
        if m isa MDP
            planner = solve(solver, m_copy)
            sim = Sim(
                m_copy,
                planner,
                rng=Xoshiro(seed + 50_000 * i),
                max_steps=sim_params.sim_max_steps,
                metadata=Dict(:solver => solver_name, :i => i, :k => solver_params.param_id)
            )
        else
            belief_mdp = GenerativeBeliefPropMDP(m_copy, node_updater)
            planner = solve(solver, belief_mdp)
            rng_filter = Xoshiro(seed + i * 90_000)
            filter = BootstrapFilter(
                m_copy,
                sim_params.num_particles_sim,
                rng_filter
            )
            sim = Sim(
                m_copy,
                planner,
                filter,
                rng=Xoshiro(seed + 50_000 * i),
                max_steps=sim_params.sim_max_steps,
                metadata=Dict(:solver => solver_name, :i => i, :k => solver_params.param_id)
            )
        end
        push!(sims, sim)
    end
    return sims
end

function gen_sims_solver(; m::Union{MDP,POMDP}, solver_params::SolverGenParams, sim_params::SimParams, seed=1234)
    sims = []
    for i in 1:sim_params.num_sims
        # Gather statistics over mean and standard deviation of the value (sum of discounted rewards)
        # For K times, run the simulation for N time steps
        if m isa POMDP
            rng_node_updater = Xoshiro(seed + 1234 + 30_000 * i)
            node_updater = BootstrapFilter(
                deepcopy(m),
                solver_params.num_particles_planner,
                rng_node_updater
            )
        end
        solver = solver_params.solver_gen(seed + 1237 + 30_000 * i)
        solver_name = solver_params.solver_name
        if m isa MDP
            planner = solve(solver, m)
            sim = Sim(
                deepcopy(m),
                planner,
                rng=Xoshiro(seed + 50_000 * i),
                max_steps=sim_params.sim_max_steps,
                metadata=Dict(:solver => solver_name * "-MDP", :i => i, :k => solver_params.param_id)
            )
        else
            belief_mdp = GenerativeBeliefPropMDP(deepcopy(m), node_updater)
            planner = solve(solver, belief_mdp)
            rng_filter = Xoshiro(seed + i * 90_000)
            filter = BootstrapFilter(
                deepcopy(m),
                sim_params.num_particles_sim,
                rng_filter
            )
            sim = Sim(
                deepcopy(m),
                planner,
                filter,
                rng=Xoshiro(seed + 50_000 * i),
                max_steps=sim_params.sim_max_steps,
                metadata=Dict(:solver => solver_name * "-POMDP", :i => i, :k => solver_params.param_id)
            )
        end
        push!(sims, sim)
    end
    return sims
end

function gen_sims(; m::Union{MDP,POMDP}, solver_params::Union{POMCPOWParams,PFTDPWParams,ActionGradMCTSParams}, sim_params::SimParams, estimate_value, next_action, default_action, seed=1234)
    if solver_params isa POMCPOWParams
        gen_func = gen_sims_pomcpow
    elseif solver_params isa PFTDPWParams
        gen_func = gen_sims_pftdpw
    elseif solver_params isa ActionGradMCTSParams
        gen_func = gen_sims_actiongradmcts
    end
    return deepcopy(gen_func(m=m, solver_params=solver_params, sim_params=sim_params, estimate_value=estimate_value, next_action=next_action, default_action=default_action, seed=seed))
end

function run_parallel_batched(process::Function, queue::AbstractVector, pool::AbstractWorkerPool=default_worker_pool(); progress=ProgressMeter.Progress(length(queue); desc="Simulating..."), proc_warn::Bool=true, show_progress::Bool=true, batch_size::Int=1)
    if nworkers(pool) == 1 && proc_warn
        @warn("""
        run_parallel(...) was started with only 1 worker in the pool, so simulations will be run in serial.

        To supress this warning, use run_parallel(..., proc_warn=false).

        To use multiple processes, use addprocs() or the -p option (e.g. julia -p 4) and make sure the correct worker pool is assigned to argument `pool` in the call to run_parallel.
        """)
    end

    if progress in (nothing, false)
        progstr = (progress === nothing) ? "nothing" : "false"
        @warn("run_parallel(..., progress=$progstr) is deprecated. Use run_parallel(..., show_progress=false) instead.")
        show_progress = false
    end

    map_function(args...; kw_args...) = (show_progress ?
                                         progress_pmap(args...; progress=progress, kw_args...) : pmap(args...; kw_args...))

    # If the simulate fails, retry with exponential backoff
    simulate_func = retry(delays=ExponentialBackOff(n=2)) do sim
        result = POMDPTools.Simulators.simulate(sim)
        output = process(sim, result)
        return merge(sim.metadata, output)
    end
    # If after all retries the simulation still fails, an error will be raised and therefore nothing will be returned
    frame_lines = map_function(simulate_func, pool, queue; batch_size=batch_size, on_error=(x) -> nothing)
    # This ensures to remove nothing from frame_lines before constructing a dataframe
    return POMDPTools.Simulators.create_dataframe(filter(x -> !isnothing(x), frame_lines))
end

function gather_sim_results(sims, parallelize_sims::Bool, process=POMDPTools.Simulators.default_process; print_results::Bool=false, batch_size::Int=1)
    """
    Gather the results of the simulations.
    """
    if parallelize_sims
        results = run_parallel_batched(process, sims; batch_size=batch_size)
    else
        results = run(process, sims)
    end
    se(vec) = std(vec) / sqrt(length(vec))
    combined_results = combine(groupby(results, :k), :reward => mean, :reward => se)

    if print_results
        @info "results:"
        @info results
        @info "combined_results:"
        @info combined_results
    end

    return results, combined_results
end

function print_params_to_file(dir_path, file_name, params)
    fpath = joinpath(dir_path, file_name)
    open(fpath, "w") do f
        JSON.print(f, params)
    end
end

function params_to_sims(init_params_container, k, param_k, lr_to_opt_func)
    # The following code could have been written much more cleanly with NamedTuple and matching based on attribute name

    if init_params_container isa POMCPOWParams
        solver_params = POMCPOWParams(
            param_id=k,
            max_query=init_params_container.max_query,
            max_tree_depth=init_params_container.max_tree_depth,
            solver_timeout=init_params_container.solver_timeout,
            ucb_c=param_k[1],
            k_act=param_k[2],
            alpha_act=param_k[3],
            k_obs=param_k[4],
            alpha_obs=param_k[5],
        )
    elseif init_params_container isa PFTDPWParams
        solver_params = PFTDPWParams(
            param_id=k,
            max_query=init_params_container.max_query,
            max_tree_depth=init_params_container.max_tree_depth,
            solver_timeout=init_params_container.solver_timeout,
            ucb_c=param_k[1],
            k_act=param_k[2],
            alpha_act=param_k[3],
            k_obs=param_k[4],
            alpha_obs=param_k[5],
            num_particles_planner=init_params_container.num_particles_planner
        )
    elseif init_params_container isa ActionGradMCTSParams
        ucb_c = init_params_container.ucb_c
        k_act = init_params_container.k_act
        alpha_act = init_params_container.alpha_act
        k_obs = init_params_container.k_obs
        alpha_obs = init_params_container.alpha_obs
        action_optimizer = init_params_container.action_optimizer
        action_update_rule = init_params_container.action_update_rule

        if length(param_k) == 1
            lr = param_k[1]
        else
            lr = param_k[6]
            ucb_c = param_k[1]
            k_act = param_k[2]
            alpha_act = param_k[3]
            k_obs = param_k[4]
            alpha_obs = param_k[5]
            if length(param_k) == 9
                action_update_rule=ActionUpdateMinChildrenEveryKMaxADist(param_k[7], param_k[8], param_k[9])
            end
        end

        action_optimizer = lr_to_opt_func(lr)

        solver_params = ActionGradMCTSParams(
            param_id=k,
            max_query=init_params_container.max_query,
            max_tree_depth=init_params_container.max_tree_depth,
            solver_timeout=init_params_container.solver_timeout,
            ucb_c=ucb_c,
            k_act=k_act,
            alpha_act=alpha_act,
            k_obs=k_obs,
            alpha_obs=alpha_obs,
            num_particles_planner=init_params_container.num_particles_planner,
            tree_in_info=init_params_container.tree_in_info,
            action_optimizer=action_optimizer,
            action_optim_iters=init_params_container.action_optim_iters,
            grad_mc_k_s=init_params_container.grad_mc_k_s,
            grad_mc_k_obs=init_params_container.grad_mc_k_obs,
            grad_mc_k_particles=init_params_container.grad_mc_k_particles,
            use_mc_immediate_reward=init_params_container.use_mc_immediate_reward,
            sample_state_reward_gradient=init_params_container.sample_state_reward_gradient,
            linearize_weight_update=init_params_container.linearize_weight_update,
            grad_weighted_by_visits=init_params_container.grad_weighted_by_visits,
            action_update_rule=action_update_rule,
            update_actions_imp_ratio_add_threshold=init_params_container.update_actions_imp_ratio_add_threshold,
            update_actions_imp_ratio_delete_threshold=init_params_container.update_actions_imp_ratio_delete_threshold,
            optimize_before_update=init_params_container.optimize_before_update,
            choose_random_obs=init_params_container.choose_random_obs,
            value_updater=init_params_container.value_updater
        )
    end
end

function cross_entropy_optimization(;
    # Problem related parameters
    prob::Union{MDP,POMDP},
    prob_name::String,
    sim_params::SimParams,
    prob_ev,
    prob_na,
    prob_da,
    # Solver related parameters
    init_params_container,
    lr_to_opt_func,
    init_mean,
    init_cov,
    fix_sampled_params_func,
    params_exp_func,
    # Optimization related parameters
    opt_max_iters=5,
    num_param_samples=20,
    num_elite_samples=10,
    score_norm::Function=(x, se) -> x,
    params_empty_csv,
    smooth_alpha::Vector{Float64}=[0.8, 0.5],
    term_eps::Float64=0.01,
    rejection_function::Union{Nothing,Function}=nothing,
    # General
    parallelize=false,
)
    datestring = Dates.format(now(), "e_d_u_Y_HH_MM")
    dir_datestring = Dates.format(now(), "dd_mm_yy")
    out_dir_path = joinpath("data", "ce_opt", prob_name, dir_datestring)
    # Make sure that directory exists
    mkpath(out_dir_path)
    out_file_suffix = solver_name_by_params(init_params_container, prob_na(1); mdp=prob isa MDP) * "_" * prob_name * "_" * datestring
    # Print simulation parameters to a file
    print_params_to_file(out_dir_path, "sim_params_" * out_file_suffix * ".json", sim_params)

    # Prepare CSV
    results_csv = empty(params_empty_csv)  # Copying column names as empty DF
    results_csv.reward_mean = Float64[]
    results_csv.reward_se = Float64[]
    results_csv.cov = Matrix{Float64}[]
    results_csv_log = empty(results_csv)  # CSV for the logspace parameters

    # Prepare parameter distribution
    params_dist_init = MvNormal(init_mean, init_cov)
    curr_dist = deepcopy(params_dist_init)
    start_mean = mean(params_dist_init)
    start_mean_length = length(start_mean)

    for i in 1:opt_max_iters
        # Run sims for each param
        sims = []
        param_samples = []
        # Sample #num_param_samples params from dist
        # Project params to allowable range
        # Apply transformation to alg param space
        while length(param_samples) < num_param_samples
            param_k = fix_sampled_params_func(rand(curr_dist))
            if rejection_function !== nothing
                if rejection_function(params_exp_func(param_k))
                    continue
                end
            end
            push!(param_samples, param_k)  # Note that param_k is in logspace, and this will get the new distribution fit to (before "exp" function)
            k = length(param_samples)
            seed = i
            param_k_exp = params_exp_func(param_k)
            solver_params_k = params_to_sims(init_params_container, k, param_k_exp, lr_to_opt_func)
            # Save parameters to a file for the first iteration - for debuggings
            if i == 1 && k == 1
                print_params_to_file(out_dir_path, "solver_params_" * out_file_suffix * ".json", solver_params_k)
            end

            curr_sims = gen_sims(m=prob,
                solver_params=solver_params_k,
                sim_params=sim_params,
                estimate_value=prob_ev,
                next_action=prob_na,
                default_action=prob_da,
                seed=seed)
            sims = vcat(sims, curr_sims)
        end

        results, combined = gather_sim_results(sims, parallelize)

        se(vec) = std(vec) / sqrt(length(vec))
        combined_by_k = combine(groupby(results, :k), :reward => mean, :reward => se)

        scores = logistic.([score_norm(a...) for a in zip(combined_by_k[!, :reward_mean], combined_by_k[!, :reward_se])])  # Converting the normalized scores to the range of (0,1)
        order = sortperm(scores)

        elite_results = combined_by_k[order[num_param_samples-num_elite_samples+1:end], :]
        elite_scores = scores[order[num_param_samples-num_elite_samples+1:end]]
        elite_samples = param_samples[combined_by_k[!, :k][order[num_param_samples-num_elite_samples+1:end]]]
        elite_matrix = Matrix{Float64}(undef, start_mean_length, num_elite_samples)
        for k in 1:num_elite_samples
            elite_matrix[:, k] = elite_samples[k]
        end
        new_dist = deepcopy(curr_dist)
        try
            new_dist = fit(typeof(curr_dist), elite_matrix, elite_scores)
        catch ex
            if ex isa PosDefException
                println(stderr, "pos def exception")
                elite_matrix = elite_matrix + randn(size(elite_matrix)) * 1e-5
                new_dist = fit(typeof(curr_dist), elite_matrix, elite_scores)
            else
                println(stderr, ex)
            end
        end
        # From a tutorial on CE optimization:
        # https://people.smp.uq.edu.au/DirkKroese/ps/CEopt.pdf
        # Smoothing parameters with 0.2 ≤ α ≤ 0.8 usually gives good and stable results.
        # Early termination criterion for continuous CE optimization is when all eigenvalues of the cov matrix are ≤ ε^2.
        new_params = [smooth_alpha[i] * getfield(new_dist, field) + (1 - smooth_alpha[i]) * getfield(curr_dist, field) for (i, field) in enumerate(fieldnames(typeof(curr_dist)))]
        curr_dist = typeof(curr_dist)(new_params...)
        curr_dist_mean = mean(curr_dist)
        curr_dist_cov = cov(curr_dist)
        curr_reward_mean = mean(elite_results[!, :reward_mean])
        curr_reward_se = mean(elite_results[!, :reward_se])
        println(stderr, "Iteration $i")
        println(stderr, "Current reward mean: ", curr_reward_mean)
        println(stderr, "Current reward se: ", curr_reward_se)
        println(stderr, "New dist mean expspace: ", params_exp_func(curr_dist_mean))
        # println(stderr, "Cov matrix logspace: ", curr_dist_cov)
        println(stderr, "Cov (det) logspace: ", det(curr_dist_cov))
        ev = eigvals(curr_dist_cov)
        println(stderr, "Cov (eig) logspace: ", ev)
        for j in 1:length(ev)
            println(stderr, "Eigvecs: ", eigvecs(curr_dist_cov)[:, j])
        end

        push!(results_csv, [params_exp_func(curr_dist_mean)..., curr_reward_mean, curr_reward_se, curr_dist_cov])
        push!(results_csv_log, [curr_dist_mean..., curr_reward_mean, curr_reward_se, curr_dist_cov])
        fname_csv = joinpath(out_dir_path, "ce_params_" * out_file_suffix * ".csv")
        fname_csv_log = joinpath(out_dir_path, "ce_log_params_" * out_file_suffix * ".csv")
        CSV.write(fname_csv, results_csv)
        CSV.write(fname_csv_log, results_csv_log)
        println(stderr, "Saving parameters to ", fname_csv)
        # Early stop condition
        if all(ev .<= term_eps^2)
            println(stderr, "Early termination at iteration $i")
            break
        end
    end
end

function compare_by_max_time_query(; m::Union{MDP,POMDP}, solvers_params, prob_name::String, parallelize::Bool=false, max_time_list=10.0 .^ (-2:0.25:0), max_query_list=[1000])
    datestring = Dates.format(now(), "e_d_u_Y_HH_MM")
    dir_datestring = Dates.format(now(), "dd_mm_yy")
    out_dir_path = joinpath("data", "time_compare", prob_name, dir_datestring)
    out_file_suffix = prob_name * "_" * datestring
    # Make sure that directory exists
    mkpath(out_dir_path)
    fname_csv = joinpath(out_dir_path, "compare" * out_file_suffix * ".csv")
    if isfile(fname_csv)
        fname_csv = joinpath(out_dir_path, "compare" * out_file_suffix * "_" * string(hash(rand())) * ".csv")
    end
    results_csv = DataFrame(k=Int[], solver_name=String[], iter_time=Float64[], max_query=Int64[], search_time_mean=Float64[], tree_queries_mean=Float64[], reward_mean=Float64[], reward_se=Float64[])

    for (i, (iter_time, max_query)) in enumerate(Iterators.product(max_time_list, max_query_list))
        sims = []
        for p in solvers_params
            new_sims = []
            params_copy = deepcopy(p[1])  # Expect p[1] to hold solver params
            params_copy.solver_timeout = iter_time
            params_copy.max_query = max_query
            sim_params = p[2]
            prob_ev = p[3]
            prob_na = p[4]
            prob_da = p[5]
            # Print initial parameters for first iteration for debugging
            if i == 1
                print_params_to_file(out_dir_path, "solver_params_" * solver_name_by_params(params_copy, prob_na(1); mdp=m isa MDP) * "_" * out_file_suffix * ".json", params_copy)
                print_params_to_file(out_dir_path, "sim_params_" * out_file_suffix * ".json", sim_params)
            end
            new_sims = gen_sims(m=m,
                solver_params=params_copy,
                sim_params=sim_params,
                estimate_value=prob_ev,
                next_action=prob_na,
                default_action=prob_da)
            sims = vcat(sims, new_sims)
        end

        function process(s::Sim, hist::SimHistory)
            df = history_table(hist)
            return (reward=discounted_reward(hist), search_time=mean(df.search_time), tree_queries=mean(df.tree_queries))
        end

        results, combined = gather_sim_results(sims, parallelize, process)
        filter_non_numbers(vec) = filter(x -> isa(x, Number) && isfinite(x), vec)
        filtered_mean(vec) = mean(filter_non_numbers(vec))
        se(vec) = std(vec) / sqrt(length(vec))
        filtered_se(vec) = se(filter_non_numbers(vec))
        combined_by_solver = combine(groupby(results, :k), :solver => first, :reward => filtered_mean => :reward_mean, :reward => filtered_se => :reward_se, :search_time => filtered_mean => :search_time_mean, :tree_queries => filtered_mean => :tree_queries_mean)

        for row in eachrow(combined_by_solver)
            result_row = [row.k, row.solver_first, iter_time, max_query, row.search_time_mean, row.tree_queries_mean, row.reward_mean, row.reward_se]
            push!(results_csv, result_row, promote=true)
        end
        CSV.write(fname_csv, results_csv)
        println(stderr, "Saving parameters to ", fname_csv)
    end
end

function plot_xva_graph(hist::SimHistory, plot_filename)
    x = [s[1] for s in hist[:s]]
    v = [s[2] for s in hist[:s]]
    a = [a for a in hist[:a]]
    idx = 1:length(x)
    gr()
    plot(idx, x, label="x(t)", xlabel="t", ylabel="x", title="Plot of x(t), v(t), a(t)")
    plot!(idx, v, label="v(t)")
    plot!(idx, a, label="a(t)")

    savefig(plot_filename)
end

"""
Prints the history table with the planner information
"""
function history_table(hist::SimHistory)
    if first(hist)[:action_info] === nothing
        out_frame = DataFrame()
    else
        action_info_keys = unique(collect(key for d in hist[:action_info] for key in keys(d)))
        out_frame = DataFrame((key => Vector{Union{Missing,Any}}(missing, length(hist[:action_info])) for key in action_info_keys)...)
        for (i, d) in enumerate(hist[:action_info])
            for (key, value) in d
                out_frame[i, key] = value
            end
        end
        action_info_out_properties = []
        for sym in [:best_Q, :tree_queries, :search_time]
            if hasproperty(out_frame, sym)
                push!(action_info_out_properties, sym)
            end
        end
        out_frame = out_frame[!, action_info_out_properties]
    end
    if hasproperty(first(hist), :b)
        out_frame[!, :belief_mean] = [mean(t[:b]) for t in hist]
        out_frame[!, :belief_cov] = [cov(t[:b]) for t in hist]
    end
    out_frame[!, :s] = [k for k in hist[:s]]  # In case we reached until here and out_frame is empty - need this first initialization for dimensions to work out.
    out_frame.a .= hist[:a]
    out_frame.sp .= hist[:sp]
    if hasproperty(hist, :o)
        out_frame.o .= hist[:o]
    end
    out_frame.r .= hist[:r]

    return out_frame
end

function print_history_table(hist::SimHistory)
    out_frame = history_table(hist)
    @info out_frame
end

function make_print_history_process(post_process=POMDPTools.Simulators.default_process)
    return function process_rewards(s::Sim, hist::SimHistory)
        print_history_table(hist)
        return post_process(s, discounted_reward(hist))
    end
end

function make_gif_process(filename_func::Function, fps=1, print_history=false, post_process=POMDPTools.Simulators.default_process)
    return function process_rewards_and_gif(s::Sim, hist::SimHistory)
        filename = filename_func(s.metadata)
        gif_filename = filename * ".gif"
        if print_history
            print_history_table(hist)
        end
        if s isa POMDPTools.Simulators.MDPSim
            POMDPGifs.makegif(s.mdp, hist; filename=gif_filename, fps=fps)
        elseif s isa POMDPTools.Simulators.POMDPSim
            POMDPGifs.makegif(s.pomdp, hist; filename=gif_filename, fps=fps)
        end
        return post_process(s, discounted_reward(hist))
    end
end

function make_pdf_process(filename_func::Function, fps=1, print_history=false, post_process=POMDPTools.Simulators.default_process)
    return function process_rewards_and_pdf(s::Sim, hist::SimHistory)
        filename = filename_func(s.metadata)
        pdf_dir_name = filename * "_pdf"
        mkpath(pdf_dir_name)

        if print_history
            print_history_table(hist)
        end

        if s isa POMDPTools.Simulators.MDPSim
            m = s.mdp
        elseif s isa POMDPTools.Simulators.POMDPSim
            m = s.pomdp
        end

        steps = eachstep(hist)
        p = Progress(length(steps); dt=0.1, desc="Rendering $(length(steps)) steps...")  # show progress

        @info "Creating PDFs..."  # show progress
        for (i, step) in enumerate(steps)
            frame = render(m, step; dpi=600)
            savefig(frame, joinpath(pdf_dir_name, "frame_$(i).pdf"))
            next!(p)  # show progress
        end
        @info "Done Creating PDFs."  # show progress

        return post_process(s, discounted_reward(hist))
    end
end

function make_gif_process_xva_plot(filename_func::Function, fps=1, print_history=false, post_process=POMDPTools.Simulators.default_process)
    return function process_rewards_and_gif(s::Sim, hist::SimHistory)
        filename = filename_func(s.metadata)
        gif_filename = filename * ".gif"
        plot_filename = filename * "_plot.png"
        if print_history
            print_history_table(hist)
        end
        plot_xva_graph(hist, plot_filename)
        if s isa POMDPTools.Simulators.MDPSim
            POMDPGifs.makegif(s.mdp, hist; filename=gif_filename, fps=fps)
        elseif s isa POMDPTools.Simulators.POMDPSim
            POMDPGifs.makegif(s.pomdp, hist; filename=gif_filename, fps=fps)
        end
        return post_process(s, discounted_reward(hist))
    end
end
