using ActionGradientMCTS
using ActionGradientMCTSExperiments
using ActionGradientMCTSExperiments.VOOSampling
using ActionGradientMCTSExperiments.VOOSampling: VOOActionGenerator
using ActionGradientMCTSExperiments.ProbMountainCar
using ActionGradientMCTSExperiments.ProbMountainCar: ProbMountainCar
using ActionGradientMCTSExperiments.LunarLander
using ActionGradientMCTSExperiments.LunarLander: LunarLander
using ActionGradientMCTSExperiments.CollabLightDark
using ActionGradientMCTSExperiments.CollabLightDark: CollabLightDark
using ArgParse, BasicPOMCP, BenchmarkTools, CSV, CovarianceEstimation, DataFrames, Dates, Distributions
using Flux, ForwardDiff, JSON, LinearAlgebra, MCTS, POMCPOW, POMDPs, POMDPTools
using Parameters, ParticleFilters, Plots, ProgressMeter, Random, StaticArrays, Statistics
using ActionGradientMCTSExperiments: compare_by_max_time_query, cross_entropy_optimization, gather_sim_results, gen_sims
using ActionGradientMCTSExperiments: SimParams, PFTDPWParams, POMCPOWParams, ActionGradMCTSParams
using ActionGradientMCTSExperiments: MDPSimIterator, POMDPSimIterator
import POMDPTools.ModelTools: GenerativeBeliefMDP, UnderlyingMDP

const use_distributed = false

function timing_tests_lunar_lander()
    mdp = LunarLanderMDP()
    pomdp = LunarLanderPOMDP(mdp=mdp)

    s = rand(initialstate(pomdp))
    a = rand(actions(pomdp))
    sp = rand(transition(pomdp, s, a))

    println("Benchmarking reward")
    @btime reward($pomdp, $s, $a, $sp)

    println("Benchmarking state grad_reward")
    @btime grad_reward($pomdp, $s, $a, $sp)

    println("Benchmarking state transition")
    @btime rand(transition($pomdp, $s, $a))

    println("Benchmarking state transition_log_likelihood")
    @btime transition_log_likelihood($pomdp, $s, $a, $sp)

    println("Benchmarking state grad_log_transition")
    @btime grad_log_transition($pomdp, $s, $a, $sp)

    println("Benchmarking observation generation")
    @btime rand(observation($pomdp, $sp))
    o = rand(observation(pomdp, sp))

    println("Benchmarking observation likelihood")
    @btime pdf(observation($pomdp, $sp), $o)

    n = 100
    rng_node_updater = Xoshiro(1234 + 30_000)
    node_updater = BootstrapFilter(deepcopy(pomdp), n, rng_node_updater)
    belief_mdp = GenerativeBeliefPropMDP(deepcopy(pomdp), node_updater)
    b = ParticleCollection([rand(initialstate(pomdp)) for i in 1:n])
    ϵ = SVector{3,Float64}(ones(3))
    bp = ParticleCollection([LunarLander.transition_f(mdp, p, a, ϵ) for p in b.particles])

    println("Benchmarking belief grad_reward")
    @btime grad_reward($belief_mdp, $b, $a, $bp)

    println("Benchmarking belief transition_log_likelihood")
    @btime transition_log_likelihood($belief_mdp, $b, $a, $bp)

    println("Benchmarking belief grad_log_transition")
    @btime grad_log_transition($belief_mdp, $b, $a, $bp)
end

function lunarlander_ev_and_na(problem, rollout_k_samples)
    lunarlander_baseline_solver = (i) -> LanderSolver()
    lunarlander_ev = (i) -> MeanStateRollout(LanderSolver(), rollout_k_samples)
    randon_action_generator = (i) -> RandomActionGenerator(Xoshiro(1242 + 30_000 * i))

    # VOO parameters
    voo_p = 0.9
    voo_exp_sampler = Distributions.Uniform()
    voo_sigs = [0.2, 0.5, 0.05]
    voo_sig = diagm(voo_sigs)
    voo_voronoi_sampler = MvNormal(voo_sig)
    voo_accept_radius_sq = 0.01^2
    voo_rng = (i) -> Xoshiro(4321 + 30_000 * i)
    voo_action_generator = (i) -> VOOActionGenerator(voo_p, voo_exp_sampler, voo_voronoi_sampler, voo_rng(i), voo_accept_radius_sq, true, 20)

    lunarlander_na_gen = (gen) -> ((i) -> PolicyFirstGen(LanderPolicy(UnderlyingMDP(deepcopy(problem))), gen(i)))
    return lunarlander_baseline_solver, lunarlander_ev, lunarlander_na_gen(randon_action_generator), lunarlander_na_gen(voo_action_generator)
end

function scenario_parameters(mdp_true_pomdp_false=false, voo=false, optim_option::Int=3)
    pomdp = LunarLanderPOMDP(;
        mdp=LunarLanderMDP(;
            # Transition parameters
            dt=0.4,
            m=1.0,
            I=10.0,
            Q=Vec6([0.0, 0.0, 0.0, 0.1, 0.1, 0.01]),
            # Action parameters
            min_lateral=-5.0,
            max_lateral=5.0,
            max_thrust=15.0,
            max_offset=1.0,
            # Reward parameters
            discount=0.99,
            max_horizontal_offset=15.0,
            max_angle=0.5,
            landed_height=1.0,
            step_penalty=-1.0,
            success_reward=100.0,
            failure_penalty=-1000.0,
            speed_step_penalty=0.0,
            offset_step_penalty=0.0,
            angle_step_penalty=0.0
        ),
        R=Vec3([1.0, 0.01, 0.1])
    )
    a_type = typeof(rand(actions(pomdp)))
    sim_max_steps = 35

    ### Solver params
    max_tree_depth = 35
    solver_timeout = 40.0
    max_query = 1000  # Copying number of queries from BOMCP paper
    grad_mc_k_s = 5
    grad_mc_k_obs = 2

    if mdp_true_pomdp_false
        num_particles_sim = 1  # Has no effect in MDP
        num_particles_planner = 1  # Has no effect in MDP
        grad_mc_k_particles = 1  # Has no effect in MDP
        rollout_k_samples = 1  # Has no effect in MDP
        lunarlander_baseline_solver, lunarlander_ev, lunarlander_na_rand, lunarlander_na_voo = lunarlander_ev_and_na(pomdp.mdp, rollout_k_samples)
    else
        num_particles_sim = 2000
        num_particles_planner = 150
        grad_mc_k_particles = 3
        rollout_k_samples = 5
        lunarlander_baseline_solver, lunarlander_ev, lunarlander_na_rand, lunarlander_na_voo = lunarlander_ev_and_na(pomdp, rollout_k_samples)
    end

    max_query_pomcpow = round(Int, max(max_query, max_query * 0.035 * num_particles_planner))

    adam_beta = (0.9, 0.999)

    if mdp_true_pomdp_false
        gradient_clip = 5000.0
    else
        gradient_clip = 80000.0
    end
    exp_decay_params = [1.0, 1.0, 1, 0.2, 0]

    if optim_option == 1
        lr_to_opt_func = (lr) -> FluxOptimizer{a_type}(Optimiser(Momentum(lr, adam_beta[1]), ExpDecay(exp_decay_params...)))
    elseif optim_option == 2
        lr_to_opt_func = (lr) -> FluxOptimizer{a_type}(Optimiser(ClipNorm(gradient_clip), Momentum(lr, adam_beta[1]), ExpDecay(exp_decay_params...)))
    elseif optim_option == 3
        lr_to_opt_func = (lr) -> FluxOptimizer{a_type}(Optimiser(Adam(lr, adam_beta), ExpDecay(exp_decay_params...)))
    elseif optim_option == 4
        lr_to_opt_func = (lr) -> FluxOptimizer{a_type}(Optimiser(ClipNorm(gradient_clip), Adam(lr, adam_beta), ExpDecay(exp_decay_params...)))
    end

    if mdp_true_pomdp_false
        if !voo
            ps_pomcpow = [101.513714168354, 9.942343584484084, 0.18220327350973564, 8.563172327068136, 0.8395219840102979]
            ps_pftdpw = [60.50410436724886, 1.4320300890600195, 0.592254533567973, 0.0699897527081039, 0.28917288553842546]
            ps_agmcts = [61.5400148214667, 1.8663018180519297, 0.5147575295820737, 0.07328065677085951, 0.9115445300756846, 0.05455979280162819, 2, 1, 1e-2]
        else
            ps_pomcpow = [70.12611523903757, 7.4376868962756815, 0.3631236093960938, 6.813571518938267, 0.7249984981381423]
            ps_pftdpw = [58.95661250905092, 1.4998635784690162, 0.5844919638770792, 0.08145332170476446, 0.6879581159112835]
            ps_agmcts = [61.09887063059996, 1.7871876896812646, 0.5346207570646173, 0.11263522099728886, 0.7410032310514396, 0.0020995735633320637, 2, 1, 1e-2]
        end

        action_optim_iters = 3
        update_actions_imp_ratio_add_threshold = 0.9
        update_actions_imp_ratio_delete_threshold = 1e-3
        use_mc_immediate_reward = false
        sample_state_reward_gradient = true
        linearize_weight_update = true
        grad_weighted_by_visits = true
        choose_random_obs = true
        action_update_rule = ActionUpdateMinChildrenEveryKMaxADist(ps_agmcts[7], ps_agmcts[8], ps_agmcts[9])
    else  # pomdp
        if !voo
            ps_pomcpow = [71.02362536844457, 2.692937502491131, 0.4554372366667195, 0.5671707145211953, 0.2831804188202277]
            ps_pftdpw = [60.87019402670538, 3.669799372580725, 0.3877933686965842, 0.24412131575230211, 0.6478673369886305]
            ps_agmcts = [50.98706853821593, 2.4075230962430063, 0.48585466310011965, 0.2782169824246397, 0.49349965534036305, 1.206202385454382e-07, 2, 1, 1e-2]
        else
            ps_pomcpow = [74.79666467847939, 3.04925400796127, 0.42593161625124265, 0.5440580320025016, 0.29396276407688854]
            ps_pftdpw = [57.802871867352835, 2.808485059661464, 0.45417508475602897, 0.21815975870910842, 0.5923346033653792]
            ps_agmcts = [50.18615462910455, 3.4698519577147966, 0.4089994804247744, 0.2784735638528604, 0.8009271874137589, 3.7332823555171853e-07, 2, 1, 1e-2]
        end

        action_optim_iters = 3
        update_actions_imp_ratio_add_threshold = 0.9
        update_actions_imp_ratio_delete_threshold = 1e-8
        use_mc_immediate_reward = true
        sample_state_reward_gradient = true
        linearize_weight_update = true
        grad_weighted_by_visits = true
        choose_random_obs = true
        action_update_rule = ActionUpdateMinChildrenEveryKMaxADist(ps_agmcts[7], ps_agmcts[8], ps_agmcts[9])
    end

    sim_params = SimParams(
        sim_max_steps=sim_max_steps,
        num_sims=1,
        num_particles_sim=num_particles_sim
    )

    baseline_solver_params = SolverGenParams(
        param_id=9,
        solver_gen=lunarlander_baseline_solver,
        solver_name="LanderSolver"
    )

    pomcpow_params = POMCPOWParams(
        param_id=1,
        max_query=max_query_pomcpow,
        max_tree_depth=max_tree_depth,
        solver_timeout=solver_timeout,
        ucb_c=ps_pomcpow[1],
        k_act=ps_pomcpow[2],
        alpha_act=ps_pomcpow[3],
        k_obs=ps_pomcpow[4],
        alpha_obs=ps_pomcpow[5],
    )

    pftdpw_params = PFTDPWParams(
        param_id=2,
        max_query=max_query,
        max_tree_depth=max_tree_depth,
        solver_timeout=solver_timeout,
        ucb_c=ps_pftdpw[1],
        k_act=ps_pftdpw[2],
        alpha_act=ps_pftdpw[3],
        k_obs=ps_pftdpw[4],
        alpha_obs=ps_pftdpw[5],
        num_particles_planner=num_particles_planner
    )

    agmcts_params = ActionGradMCTSParams(
        param_id=3,
        max_query=max_query,
        max_tree_depth=max_tree_depth,
        solver_timeout=solver_timeout,
        ucb_c=ps_agmcts[1],
        k_act=ps_agmcts[2],
        alpha_act=ps_agmcts[3],
        k_obs=ps_agmcts[4],
        alpha_obs=ps_agmcts[5],
        num_particles_planner=num_particles_planner,
        action_optimizer=lr_to_opt_func(ps_agmcts[6]),
        action_optim_iters=action_optim_iters,
        grad_mc_k_s=grad_mc_k_s,
        grad_mc_k_obs=grad_mc_k_obs,
        grad_mc_k_particles=grad_mc_k_particles,
        use_mc_immediate_reward=use_mc_immediate_reward,
        sample_state_reward_gradient=sample_state_reward_gradient,
        linearize_weight_update=linearize_weight_update,
        grad_weighted_by_visits=grad_weighted_by_visits,
        action_update_rule=action_update_rule,
        update_actions_imp_ratio_add_threshold=update_actions_imp_ratio_add_threshold,
        update_actions_imp_ratio_delete_threshold=update_actions_imp_ratio_delete_threshold,
        optimize_before_update=true,
        choose_random_obs=choose_random_obs,
        value_updater=ValueUpdaterSNMISMC()
    )

    lunarlander_default_action = (i) -> BasicPOMCP.ReportWhenUsed(Vec3(zeros(3)))
    # lunarlander_default_action = (i) -> BasicPOMCP.ExceptionRethrow()

    return pomdp, a_type, sim_params, baseline_solver_params, ps_pomcpow, pomcpow_params, ps_pftdpw, pftdpw_params, ps_agmcts, agmcts_params, lr_to_opt_func, rollout_k_samples, lunarlander_ev, lunarlander_na_rand, lunarlander_na_voo, lunarlander_default_action
end

function simulation_lunarlander_pomdp(mdp_true_pomdp_false=false; solver="ag-dpw")
    ### Parameter objects initialization
    voo = solver_uses_voo(solver)
    pomdp, a_type, sim_params, baseline_solver_params, ps_pomcpow, pomcpow_params, ps_pftdpw, pftdpw_params, ps_agmcts, agmcts_params, lr_to_opt_func, rollout_k_samples, lunarlander_ev, lunarlander_na_rand, lunarlander_na_voo, lunarlander_default_action = scenario_parameters(mdp_true_pomdp_false, voo)

    p = mdp_true_pomdp_false ? pomdp.mdp : pomdp

    agmcts_params.tree_in_info = true

    sim_params_warmup = deepcopy(sim_params)
    sim_params_warmup.num_sims = use_distributed ? num_workers : 1
    sim_params.num_sims = use_distributed ? 20 : 2

    out_name_init = "gifs/LunarLander/LunarLanderPOMDP/LunarLanderPOMDP"
    out_filename = (metadata) -> out_name_init * "$(metadata[:k])_$(metadata[:i])"
    # gif_process = make_gif_process(out_filename, 1, true)
    # process = make_gif_process(out_filename, 25, true)
    # process = make_gif_process_xva_plot(out_filename, 25, true)
    # process = D == 2 ? make_pdf_process(out_filename, 1, true) : make_print_history_process()
    process = make_print_history_process()

    @info "warmup for compilation"
    sims_baseline = gen_sims_solver(m=p,
        solver_params=baseline_solver_params,
        sim_params=sim_params_warmup
    )
    gather_sim_results(sims_baseline, use_distributed, process; print_results=true)
    if !mdp_true_pomdp_false && solver_is_pomcpow(solver)
        sims_pomcpow = gen_sims(m=p,
            solver_params=pomcpow_params,
            sim_params=sim_params_warmup,
            estimate_value=lunarlander_ev,
            next_action=voo ? lunarlander_na_voo : lunarlander_na_rand,
            default_action=lunarlander_default_action
        )
        gather_sim_results(sims_pomcpow, use_distributed, process; print_results=true)
    end
    if solver_is_dpw(solver)
        sims_pftdpw = gen_sims(m=p,
            solver_params=pftdpw_params,
            sim_params=sim_params_warmup,
            estimate_value=lunarlander_ev,
            next_action=voo ? lunarlander_na_voo : lunarlander_na_rand,
            default_action=lunarlander_default_action
        )
        gather_sim_results(sims_pftdpw, use_distributed, process; print_results=true)
    end
    if solver_is_agmcts(solver)
        sims_agmcts = gen_sims(m=p,
            solver_params=agmcts_params,
            sim_params=sim_params_warmup,
            estimate_value=lunarlander_ev,
            next_action=voo ? lunarlander_na_voo : lunarlander_na_rand,
            default_action=lunarlander_default_action
        )
        gather_sim_results(sims_agmcts, use_distributed, process; print_results=true)
    end

    if !mdp_true_pomdp_false && solver_is_pomcpow(solver)
        @info "Running simulations LunarLander $(canonical_solver_name(solver; mdp=mdp_true_pomdp_false))"
        sims_pomcpow = gen_sims(m=p,
            solver_params=pomcpow_params,
            sim_params=sim_params,
            estimate_value=lunarlander_ev,
            next_action=voo ? lunarlander_na_voo : lunarlander_na_rand,
            default_action=lunarlander_default_action
        )
        gather_sim_results(sims_pomcpow, use_distributed, process; print_results=true)
    end

    if solver_is_dpw(solver)
        @info "Running simulations LunarLander $(canonical_solver_name(solver; mdp=mdp_true_pomdp_false))"
        sims_pftdpw = gen_sims(m=p,
            solver_params=pftdpw_params,
            sim_params=sim_params,
            estimate_value=lunarlander_ev,
            next_action=voo ? lunarlander_na_voo : lunarlander_na_rand,
            default_action=lunarlander_default_action
        )
        gather_sim_results(sims_pftdpw, use_distributed, process; print_results=true)
    end

    if solver_is_agmcts(solver)
        @info "Running simulations LunarLander $(canonical_solver_name(solver; mdp=mdp_true_pomdp_false))"
        sims_agmcts = gen_sims(m=p,
            solver_params=agmcts_params,
            sim_params=sim_params,
            estimate_value=lunarlander_ev,
            next_action=voo ? lunarlander_na_voo : lunarlander_na_rand,
            default_action=lunarlander_default_action
        )
        gather_sim_results(sims_agmcts, use_distributed, process; print_results=true)
    end

    return
end

function ce_opt_lunarlander(; mdp_true_pomdp_false=false, solver="ag-dpw", optim_option::Int=3, test_mode=false)
    opt_iters = 50
    num_param_samples = 150
    num_elite_samples = 30
    smooth_alpha = [0.8, 0.5]
    score_norm_lunarlander = (x, se) -> (x - se - 55) / 10  # Linear mapping from [45, 65] to [-1,1]

    voo = solver_uses_voo(solver)
    pomdp, a_type, sim_params, baseline_solver_params, ps_pomcpow, pomcpow_params, ps_pftdpw, pftdpw_params, ps_agmcts, agmcts_params, lr_to_opt_func, rollout_k_samples, lunarlander_ev, lunarlander_na_rand, lunarlander_na_voo, lunarlander_default_action = scenario_parameters(mdp_true_pomdp_false, voo, optim_option)

    sim_params.num_sims = use_distributed ? 40 : 2

    if test_mode
        opt_iters = 2
        num_param_samples = 12
        num_elite_samples = 10
        sim_params.num_sims = 2
    end

    prob = mdp_true_pomdp_false ? pomdp.mdp : pomdp
    prob_name = "LunarLander" * (mdp_true_pomdp_false ? "MDP" : "POMDP")

    ### POMCPOW params
    init_params_container_pomcpow = pomcpow_params
    params_empty_csv_pomcpow = DataFrame(ucb_c=Float64[], k_act=Float64[], alpha_act=Float64[], k_obs=Float64[], alpha_obs=Float64[])
    params_log_func_pomcpow = (p) -> p
    params_exp_func_pomcpow = (p) -> p
    init_mean_pomcpow = ps_pomcpow
    init_logmean_pomcpow = params_log_func_pomcpow(init_mean_pomcpow)
    init_cov_pomcpow = diagm([40.0, 3.0, 0.5, 3.0, 0.5] .^ 2)
    fix_sampled_params_func_pomcpow = (p) -> [max(p[1], 0), max(p[2], 0), clamp(p[3], 0, 1), max(p[4], 0), clamp(p[5], 0, 1)]

    ### PFT-DPW params
    init_params_container_pftdpw = pftdpw_params
    params_empty_csv_pftdpw = deepcopy(params_empty_csv_pomcpow)
    params_log_func_pftdpw = params_log_func_pomcpow
    params_exp_func_pftdpw = params_exp_func_pomcpow
    init_mean_pftdpw = ps_pftdpw
    init_logmean_pftdpw = params_log_func_pftdpw(init_mean_pftdpw)
    init_cov_pftdpw = deepcopy(init_cov_pomcpow)
    fix_sampled_params_func_pftdpw = fix_sampled_params_func_pomcpow

    ### AGMCTS params
    init_params_container_agmcts = agmcts_params
    params_empty_csv_agmcts = DataFrame(ucb_c=Float64[], k_act=Float64[], alpha_act=Float64[], k_obs=Float64[], alpha_obs=Float64[], lr=Float64[])
    params_log_func_agmcts = (p) -> [p[1] / 10.0, p[2], p[3] * 10.0, p[4], p[5] * 10.0, log(p[6])]
    params_exp_func_agmcts = (p) -> [p[1] * 10.0, p[2], p[3] / 10.0, p[4], p[5] / 10.0, exp(p[6])]
    init_mean_agmcts = ps_agmcts[1:6]
    init_logmean_agmcts = params_log_func_agmcts(init_mean_agmcts)
    init_cov_agmcts = diagm([8.0, 3.0, 4.0, 3.0, 4.0, 4.0] .^ 2)
    fix_sampled_params_func_agmcts = (p) -> [max(p[1], 0.0), max(p[2], 0.0), clamp(p[3], 0.0, 10.0), max(p[4], 0.0), clamp(p[5], 0.0, 10.0), min(p[6], 0.0)]
    rejection_function_agmcts = (p) -> begin
        max_query = agmcts_params.max_query
        optim_iters = agmcts_params.action_optim_iters
        k_act = p[2]
        k_obs = p[4]
        alpha_obs = p[5]
        step_size = p[6]
        min_children = ps_agmcts[7]
        k_visits = ps_agmcts[8]
        max_a_dist = ps_agmcts[9]
        if min_children == 1
            gradient_start_visits = 1
        else
            gradient_start_visits = ceil(Int, clamp((min_children / (k_obs + 1e-4))^(1 / (alpha_obs + 1e-4)), 0.0, max_query))
        end
        gradient_count_visits = (optim_iters / k_visits) * (max_query - gradient_start_visits - k_act)
        return gradient_count_visits * step_size < 0.01 * max_a_dist
    end

    ## PFT-DPW
    if solver_is_dpw(solver)
        cross_entropy_optimization(;
            # Problem related parameters
            prob=prob,
            prob_name=prob_name,
            sim_params=sim_params,
            prob_ev=lunarlander_ev,
            prob_na=voo ? lunarlander_na_voo : lunarlander_na_rand,
            prob_da=lunarlander_default_action,
            # Solver related parameters
            init_params_container=init_params_container_pftdpw,
            lr_to_opt_func=lr_to_opt_func,
            init_mean=init_logmean_pftdpw,
            init_cov=init_cov_pftdpw,
            fix_sampled_params_func=fix_sampled_params_func_pftdpw,
            params_exp_func=params_exp_func_pftdpw,
            # Optimization related parameters
            opt_max_iters=opt_iters,
            num_param_samples=num_param_samples,
            num_elite_samples=num_elite_samples,
            score_norm=score_norm_lunarlander,
            params_empty_csv=params_empty_csv_pftdpw,
            smooth_alpha=smooth_alpha,
            # term_eps::Float64=0.1,
            # General
            parallelize=use_distributed,
        )
    end

    ## POMCPOW
    if solver_is_pomcpow(solver) && !mdp_true_pomdp_false
        cross_entropy_optimization(;
            # Problem related parameters
            prob=prob,
            prob_name=prob_name,
            sim_params=sim_params,
            prob_ev=lunarlander_ev,
            prob_na=voo ? lunarlander_na_voo : lunarlander_na_rand,
            prob_da=lunarlander_default_action,
            # Solver related parameters
            init_params_container=init_params_container_pomcpow,
            lr_to_opt_func=lr_to_opt_func,
            init_mean=init_logmean_pomcpow,
            init_cov=init_cov_pomcpow,
            fix_sampled_params_func=fix_sampled_params_func_pomcpow,
            params_exp_func=params_exp_func_pomcpow,
            # Optimization related parameters
            opt_max_iters=opt_iters,
            num_param_samples=num_param_samples,
            num_elite_samples=num_elite_samples,
            score_norm=score_norm_lunarlander,
            params_empty_csv=params_empty_csv_pomcpow,
            smooth_alpha=smooth_alpha,
            # term_eps::Float64=0.1,
            # General
            parallelize=use_distributed,
        )
    end

    ## AGMCTS
    if solver_is_agmcts(solver)
        cross_entropy_optimization(;
            # Problem related parameters
            prob=prob,
            prob_name=prob_name,
            sim_params=sim_params,
            prob_ev=lunarlander_ev,
            prob_na=voo ? lunarlander_na_voo : lunarlander_na_rand,
            prob_da=lunarlander_default_action,
            # Solver related parameters
            init_params_container=init_params_container_agmcts,
            lr_to_opt_func=lr_to_opt_func,
            init_mean=init_logmean_agmcts,
            init_cov=init_cov_agmcts,
            fix_sampled_params_func=fix_sampled_params_func_agmcts,
            params_exp_func=params_exp_func_agmcts,
            # Optimization related parameters
            opt_max_iters=opt_iters,
            num_param_samples=num_param_samples,
            num_elite_samples=num_elite_samples,
            score_norm=score_norm_lunarlander,
            params_empty_csv=params_empty_csv_agmcts,
            smooth_alpha=smooth_alpha,
            # term_eps::Float64=0.1,
            rejection_function=rejection_function_agmcts,
            # General
            parallelize=use_distributed,
        )
    end
end

function max_time_lunarlander(; mdp_true_pomdp_false=false, solver="ag-dpw", optim_option::Int=3, test_mode=false)
    voo = solver_uses_voo(solver)
    pomdp, a_type, sim_params, baseline_solver_params, ps_pomcpow, pomcpow_params, ps_pftdpw, pftdpw_params, ps_agmcts, agmcts_params, lr_to_opt_func, rollout_k_samples, lunarlander_ev, lunarlander_na_rand, lunarlander_na_voo, lunarlander_default_action = scenario_parameters(mdp_true_pomdp_false, voo, optim_option)

    max_time_list = [30.0]
    max_query = 1000
    max_query_list = map(x -> round(Int, max_query * x), 10.0 .^ (-1.0:0.25:0.0))

    max_query_list_pomcpow = [round(Int, nq * sqrt(pftdpw_params.num_particles_planner) * 0.5 / 0.7) for nq in max_query_list]  # Empirically tuned
    sim_params.num_sims = test_mode ? 10 : 1000

    prob = mdp_true_pomdp_false ? pomdp.mdp : pomdp
    prob_name = "LunarLander" * (mdp_true_pomdp_false ? "MDP" : "POMDP")

    if !mdp_true_pomdp_false && solver_is_pomcpow(solver)
        compare_by_max_time_query(; m=prob,
            solvers_params=[[pomcpow_params, sim_params, lunarlander_ev, voo ? lunarlander_na_voo : lunarlander_na_rand, lunarlander_default_action]],
            prob_name=prob_name,
            parallelize=use_distributed,
            max_time_list=max_time_list,
            max_query_list=max_query_list_pomcpow
        )
    end

    if solver_is_dpw(solver)
        compare_by_max_time_query(; m=prob,
            solvers_params=[[pftdpw_params, sim_params, lunarlander_ev, voo ? lunarlander_na_voo : lunarlander_na_rand, lunarlander_default_action]],
            prob_name=prob_name,
            parallelize=use_distributed,
            max_time_list=max_time_list,
            max_query_list=max_query_list
        )
    end

    if solver_is_agmcts(solver)
        compare_by_max_time_query(; m=prob,
            solvers_params=[[agmcts_params, sim_params, lunarlander_ev, voo ? lunarlander_na_voo : lunarlander_na_rand, lunarlander_default_action]],
            prob_name=prob_name,
            parallelize=use_distributed,
            max_time_list=max_time_list,
            max_query_list=max_query_list
        )
    end
end

"""
    run_sim(args)

Run a simulation based on parsed arguments.
"""
function run_sim(args)
    println("Running simulation...")
    for solver in args["solvers"]
        simulation_lunarlander_pomdp(args["mdp"];
            solver=solver
        )
    end
end

"""
    run_ce_opt(args)

Run the CE optimization experiments based on parsed arguments.
"""
function run_ce_opt(args)
    println("Running CE optimization...")
    for solver in args["solvers"]
        ce_opt_lunarlander(
            mdp_true_pomdp_false=args["mdp"],
            solver=solver,
            test_mode=args["test-mode"]
        )
    end
end

"""
    run_ablation(args)

Run the ablation experiments based on parsed arguments.
"""
function run_ablation(args)
    println("Running ablation...")
    for solver in args["solvers"]
        max_time_lunarlander(
            mdp_true_pomdp_false=args["mdp"],
            solver=solver,
            test_mode=args["test-mode"]
        )
    end
end

"""
Main entry point. Parses arguments and calls the appropriate function.
"""
function main(parsed_args)
    if parsed_args["sim"]
        run_sim(parsed_args)
    end
    if parsed_args["ce-opt"]
        run_ce_opt(parsed_args)
    end
    if parsed_args["ablation"]
        run_ablation(parsed_args)
    end
end
