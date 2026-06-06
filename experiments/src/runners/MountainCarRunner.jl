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

function timing_tests_prob_mountain_car(simple_true_ode_false=true)
    if simple_true_ode_false
        mdp = SimpleMountainCarMDP()
    else
        mdp = ODEMountainCarMDP()
    end
    pomdp = ProbMountainCarPOMDP(mdp=mdp)

    s = rand(initialstate(pomdp))
    a = rand(actions(pomdp))
    sp = rand(transition(pomdp, s, a))

    println("Benchmarking reward")
    @btime reward($pomdp, $s, $a, $sp)

    println("Benchmarking state grad_reward")
    @btime grad_reward($pomdp, $s, $a, $sp)

    println("Benchmarking state transition")
    rand(transition(pomdp, s, a))
    @btime rand(transition($pomdp, $s, $a))

    println("Benchmarking state transition_log_likelihood")
    transition_log_likelihood(pomdp, s, a, sp)
    @btime transition_log_likelihood($pomdp, $s, $a, $sp)

    println("Benchmarking state grad_log_transition")
    grad_log_transition(pomdp, s, a, sp)
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
    w = 0.1
    bp = ParticleCollection([rand(transition(belief_mdp.gmdp.pomdp, p, a)) for p in b.particles])

    println("Benchmarking belief grad_reward")
    grad_reward(belief_mdp, b, a, bp)
    @btime grad_reward($belief_mdp, $b, $a, $bp)

    println("Benchmarking belief transition_log_likelihood")
    transition_log_likelihood(belief_mdp, b, a, bp)
    @btime transition_log_likelihood($belief_mdp, $b, $a, $bp)

    println("Benchmarking belief grad_log_transition")
    grad_log_transition(belief_mdp, b, a, bp)
    @btime grad_log_transition($belief_mdp, $b, $a, $bp)

    nothing
end

function compare_belief_transition_weight_mc(simple_true_ode_false=true)
    if simple_true_ode_false
        mdp = SimpleMountainCarMDP()
    else
        mdp = ODEMountainCarMDP()
    end
    pomdp = ProbMountainCarPOMDP(mdp=mdp)

    s = rand(initialstate(pomdp))
    a = rand(actions(pomdp))
    sp = rand(transition(pomdp, s, a))


    particle_counts = [50, 100, 200, 400, 800]
    ks = [("0.5*p", (p) -> 0.5 * p),
        ("0.8*p", (p) -> 0.8 * p),
        ("0.95*p", (p) -> 0.95 * p),
        ("0.99*p", (p) -> 0.99 * p),
    ]

    a_prime = a + 0.001 * pomdp.mdp.action_max

    for pcount in particle_counts
        @info "particle count: $pcount"

        rng_node_updater = Xoshiro(1234 + 30_000)
        node_updater = BootstrapFilter(pomdp, pcount, rng_node_updater)
        belief_mdp = GenerativeBeliefPropMDP(pomdp, node_updater)
        b = ParticleCollection([rand(initialstate(pomdp)) for i in 1:pcount])
        bp = ParticleCollection([rand(transition(belief_mdp.gmdp.pomdp, p, a)) for p in b.particles])

        analytic_qi = ActionGradientMCTS.transition_log_likelihood(belief_mdp, b, a, bp)
        analytic_pi = ActionGradientMCTS.transition_log_likelihood(belief_mdp, b, a_prime, bp)
        analytic_log_ratio = analytic_pi - analytic_qi
        analytic_weight = exp(analytic_log_ratio)
        @info "analytic_log_ratio: $analytic_log_ratio, analytic_weight: $analytic_weight"

        for k_tuple in ks
            k_name, k_func = k_tuple
            k = round(Int, k_func(pcount))
            if k > pcount || k < 1
                continue
            end
            @info "k_func: $k_name, k samples: $k"
            mc_log_ratios = []
            mc_weights = []
            for j in 1:1000
                sampled_indices = sample(1:pcount, k, replace=true)
                mc_qi = @views reduce(+, map((i) -> ActionGradientMCTS.transition_log_likelihood(belief_mdp.gmdp.pomdp, particle(b, i), a, particle(bp, i)), sampled_indices)) * (pcount / k)
                mc_pi = @views reduce(+, map((i) -> ActionGradientMCTS.transition_log_likelihood(belief_mdp.gmdp.pomdp, particle(b, i), a_prime, particle(bp, i)), sampled_indices)) * (pcount / k)
                mc_log_ratio = mc_pi - mc_qi
                mc_weight = exp(mc_log_ratio)
                push!(mc_log_ratios, mc_log_ratio)
                push!(mc_weights, mc_weight)
            end
            mc_log_ratio = mean(mc_log_ratios)
            mc_weight = mean(mc_weights)
            bias_log_ratio = mc_log_ratio - analytic_log_ratio
            bias_weight = mc_weight - analytic_weight
            variance_log_ratio = var(mc_log_ratios)
            variance_weight = var(mc_weights)
            @info "mc_log_ratio: $mc_log_ratio, bias_log_ratio: $bias_log_ratio, variance_log_ratio: $variance_log_ratio"
            @info "mc_weight: $mc_weight, bias_weight: $bias_weight, variance_weight: $variance_weight"
        end
    end
end

function compare_belief_transition_weight_linearized(simple_true_ode_false=true)
    if simple_true_ode_false
        mdp = SimpleMountainCarMDP()
    else
        mdp = ODEMountainCarMDP()
    end
    pomdp = ProbMountainCarPOMDP(mdp=mdp)

    s = rand(initialstate(pomdp))
    a = rand(actions(pomdp))
    sp = rand(transition(pomdp, s, a))


    particle_counts = [20, 40, 50, 100, 200]
    ks = [("1", (p) -> 1),
        ("2", (p) -> 2),
        ("4", (p) -> 4),
        ("5", (p) -> 5),
        ("10", (p) -> 10),
    ]
    ϵ = 1e-3
    δa = ϵ * pomdp.mdp.action_std
    a_prime = a + δa

    for pcount in particle_counts
        @info "particle count: $pcount"

        rng_node_updater = Xoshiro(1234 + 30_000)
        node_updater = BootstrapFilter(pomdp, pcount, rng_node_updater)
        belief_mdp = GenerativeBeliefPropMDP(pomdp, node_updater)
        b = ParticleCollection([rand(initialstate(pomdp)) for i in 1:pcount])
        bp = ParticleCollection([rand(transition(belief_mdp.gmdp.pomdp, p, a)) for p in b.particles])

        analytic_qi = ActionGradientMCTS.transition_log_likelihood(belief_mdp, b, a, bp)
        analytic_pi = ActionGradientMCTS.transition_log_likelihood(belief_mdp, b, a_prime, bp)
        analytic_log_ratio = analytic_pi - analytic_qi
        analytic_weight = exp(analytic_log_ratio)
        @info "analytic_log_ratio: $analytic_log_ratio, analytic_weight: $analytic_weight"

        for k_tuple in ks
            k_name, k_func = k_tuple
            k = round(Int, k_func(pcount))
            if k > pcount || k < 1
                continue
            end
            @info "k_func: $k_name, k samples: $k"
            mc_log_ratios = []
            mc_weights = []
            for j in 1:100000
                sampled_indices = sample(1:pcount, k, replace=true)
                grad_pi = @views reduce(+, map((i) -> ActionGradientMCTS.grad_log_transition(belief_mdp.gmdp.pomdp, particle(b, i), a, particle(bp, i)), sampled_indices)) * (pcount / k)
                mc_log_ratio = grad_pi' * δa # The starting log_ratio is 0
                mc_weight = exp(mc_log_ratio)
                push!(mc_log_ratios, mc_log_ratio)
                push!(mc_weights, mc_weight)
            end
            mc_log_ratio = mean(mc_log_ratios)
            mc_weight = mean(mc_weights)
            rel_bias_log_ratio = (mc_log_ratio - analytic_log_ratio) / analytic_log_ratio
            rel_bias_weight = (mc_weight - analytic_weight) / analytic_weight
            rel_std_log_ratio = sqrt(var(mc_log_ratios)) / analytic_log_ratio  # .- analytic_log_ratio
            rel_std_weight = sqrt(var(mc_weights)) / analytic_weight  # .- analytic_weight
            @info "mc_log_ratio: $mc_log_ratio, rel_bias_log_ratio: $rel_bias_log_ratio, rel_std_log_ratio: $rel_std_log_ratio"
            @info "mc_weight: $mc_weight, rel_bias_weight: $rel_bias_weight, rel_std_weight: $rel_std_weight"
        end
    end
end

function probmountaincar_ev_and_na(problem, rollout_k_samples, threshold_vel=0.0)
    mountaincar_baseline_solver = (i) -> ThresholdVSolver(threshold=threshold_vel, rng=Xoshiro(1235 + 30_000 * i))
    mountaincar_ev = (i) -> MeanStateRollout(ThresholdVSolver(threshold=threshold_vel, rng=Xoshiro(1235 + 30_000 * i)), rollout_k_samples)

    randon_action_generator = (i) -> RandomActionGenerator(Xoshiro(1242 + 30_000 * i))

    # VOO parameters
    voo_p = 0.85
    voo_exp_sampler = Distributions.Uniform()
    voo_sig = 0.05
    voo_voronoi_sampler = Normal(voo_sig)
    voo_accept_radius_sq = 0.01
    voo_rng = (i) -> Xoshiro(4321 + 30_000 * i)
    voo_action_generator = (i) -> VOOActionGenerator(voo_p, voo_exp_sampler, voo_voronoi_sampler, voo_rng(i), voo_accept_radius_sq, true, 20)

    mountaincar_na_gen = (gen) -> ((i) -> PolicyFirstGen(ThresholdVPolicy(prob=UnderlyingMDP(deepcopy(problem)), threshold=threshold_vel, rng=Xoshiro(1235 + 30_000 * i)), gen(i)))
    return mountaincar_baseline_solver, mountaincar_ev, mountaincar_na_gen(randon_action_generator), mountaincar_na_gen(voo_action_generator)
end

function scenario_parameters(mdp_true_pomdp_false=false, simple_true_ode_false=true, voo=false, optim_option::Int=3)
    if simple_true_ode_false
        mdp = SimpleMountainCarMDP(;
            x_min=-1.5,
            x_max=0.5,
            v_min=-0.05,
            v_max=0.05,
            action_min=-1.0,
            action_max=1.0,
            action_std=0.1,
            # Simplified transition parameters
            mountain_coeff=3.0,
            a_to_v_coeff=0.001,
            x_to_v_coeff=-0.0025,
            # Reward parameters
            reward_goal=100.0,
            velocity_goal_penalty=0.0,
            distance_from_goal_penalty=0.0,
            reward_step=-0.1,
            failure_penalty=-100.0,
            discount=0.99,
            action_penalty_coeff=-0.0
        )
    else
        mdp = ODEMountainCarMDP(;
            x_min=-1.0,
            x_max=1.0,
            v_min=-2.5,
            v_max=2.5,
            action_min=-4.0,
            action_max=4.0,
            action_std=0.1,
            # ODE transition parameters
            hill_car_mass=1.0,
            hill_gravity=9.81,
            hill_integration_dt=0.01,  # time interval for numerical integration
            hill_time_step=0.1,  # time interval between consecutive observations and actions
            # Reward parameters
            reward_goal=100.0,
            velocity_goal_penalty=0.0,
            distance_from_goal_penalty=0.0,
            reward_step=-1.0,
            failure_penalty=-100.0,
            discount=0.99,
            action_penalty_coeff=-0.0
        )
    end

    pomdp = ProbMountainCarPOMDP(;
        mdp=mdp,
        meas_std=0.03
    )
    a_type = typeof(rand(actions(pomdp)))

    if simple_true_ode_false
        sim_max_steps = 200
        max_tree_depth = 200
    else
        sim_max_steps = 30
        max_tree_depth = 30
    end

    ### Solver params
    solver_timeout = 40.0
    max_query = 500

    grad_mc_k_s = 5
    grad_mc_k_obs = 2
    baseline_solver_threshold_vel = 0.0

    if mdp_true_pomdp_false
        num_particles_sim = 1
        num_particles_planner = 1
        grad_mc_k_particles = 1
        rollout_k_samples = 1
        mountaincar_baseline_solver, mountaincar_ev, mountaincar_na_rand, mountaincar_na_voo = probmountaincar_ev_and_na(pomdp.mdp, rollout_k_samples, baseline_solver_threshold_vel)
    else
        num_particles_sim = 200
        num_particles_planner = 30
        grad_mc_k_particles = 3
        rollout_k_samples = 5
        mountaincar_baseline_solver, mountaincar_ev, mountaincar_na_rand, mountaincar_na_voo = probmountaincar_ev_and_na(pomdp, rollout_k_samples, baseline_solver_threshold_vel)
    end

    max_query_pomcpow = round(Int, max(max_query, max_query * 0.08 * num_particles_planner))

    adam_beta = (0.9, 0.999)

    # clip value was chosen as 10x of observed median gradient norm
    if mdp_true_pomdp_false
        if simple_true_ode_false
            gradient_clip = 1.5
        else
            gradient_clip = 0.5
        end
    else
        if simple_true_ode_false
            gradient_clip = 80.0
        else
            gradient_clip = 5e-12  # This seems really weird
        end
    end

    exp_decay_params = [1.0, 1.0, 1, 0.5, 0]

    if optim_option == 1
        lr_to_opt_func = (lr) -> FluxOptimizer{a_type}(Optimiser(Momentum(lr, adam_beta[1]), ExpDecay(exp_decay_params...)))
    elseif optim_option == 2
        lr_to_opt_func = (lr) -> FluxOptimizer{a_type}(Optimiser(ClipNorm(gradient_clip), Momentum(lr, adam_beta[1]), ExpDecay(exp_decay_params...)))
    elseif optim_option == 3
        lr_to_opt_func = (lr) -> FluxOptimizer{a_type}(Optimiser(Adam(lr, adam_beta), ExpDecay(exp_decay_params...)))
    elseif optim_option == 4
        lr_to_opt_func = (lr) -> FluxOptimizer{a_type}(Optimiser(ClipNorm(gradient_clip), Adam(lr, adam_beta), ExpDecay(exp_decay_params...)))
    end

    if simple_true_ode_false && mdp_true_pomdp_false
        if !voo
            ps_pomcpow = [61.26878273753814, 2.322423752404812, 0.7575599638931165, 0.40542209786838124, 0.4023718684687815]
            ps_pftdpw = [112.19604737560566, 6.129930786696953, 0.5950082263551097, 0.24034446053109704, 0.36307772814928896]
            ps_agmcts = [0.0, 5.016184503670573, 0.6746168760614861, 0.1979323027752925, 0.5670002214218739, 0.0003969264311264018, 2, 1, 1.0 * mdp.action_std]
        else
            ps_pomcpow = [61.26878273753814, 2.322423752404812, 0.7575599638931165, 0.40542209786838124, 0.4023718684687815]
            ps_pftdpw = [116.80496275539379, 2.0933489795280367, 0.7175158188887224, 0.2784029318360075, 0.6189454391288486]
            ps_agmcts = [39.90337689611615, 9.07532205732198, 0.02317178810592283, 3.379084307889002, 0.5357249274520144, 0.10958341408183753, 2, 1, 1.0 * mdp.action_std]
        end

        action_optim_iters = 3
        update_actions_imp_ratio_add_threshold = 1.0
        update_actions_imp_ratio_delete_threshold = 0.5
        use_mc_immediate_reward = false
        sample_state_reward_gradient = true
        grad_weighted_by_visits = false
        choose_random_obs = true
        action_update_rule = ActionUpdateMinChildrenEveryKMaxADist(ps_agmcts[7], ps_agmcts[8], ps_agmcts[9])
    elseif simple_true_ode_false && !mdp_true_pomdp_false
        if !voo
            ps_pomcpow = [144.4557447500487, 3.9306581416633333, 0.6444549944737732, 0.4660376590964002, 0.09764410517625642]
            ps_pftdpw = [119.06218487400463, 5.397718489987546, 0.8677817573012694, 0.9152930977501816, 0.4792410124835225]
            ps_agmcts = [43.364886602576014, 3.1126329445604317, 0.026441447342351005, 5.68561400123871, 0.4622691431741421, 0.016833266572152207, 2, 1, 1.0 * mdp.action_std]
        else
            ps_pomcpow = [70.89349510683897, 5.234934973984675, 0.6711426346729477, 0.3721632872359978, 0.47654280344596933]
            ps_pftdpw = [139.54439061424472, 8.960027300036176, 0.7561320029753535, 3.22791507075703, 0.29825369475451446]
            ps_agmcts = [46.96537269151654, 3.1037667220047322, 0.02915896171571087, 5.633236946360143, 0.5807220949108748, 0.013638758775645328, 2, 1, 1.0 * mdp.action_std]
        end

        action_optim_iters = 3
        update_actions_imp_ratio_add_threshold = 0.99
        update_actions_imp_ratio_delete_threshold = 1e-8
        use_mc_immediate_reward = true
        sample_state_reward_gradient = true
        grad_weighted_by_visits = false
        choose_random_obs = true
        action_update_rule = ActionUpdateMinChildrenEveryKMaxADist(ps_agmcts[7], ps_agmcts[8], ps_agmcts[9])
    elseif !simple_true_ode_false && mdp_true_pomdp_false
        if !voo
            ps_pomcpow = [144.4557447500487, 3.9306581416633333, 0.6444549944737732, 0.4660376590964002, 0.09764410517625642]
            ps_pftdpw = [177.9894508344407, 6.732709436525945, 0.6169993744038026, 0.5191436427454263, 0.2595976627870204]
            ps_agmcts = [169.9175369797921, 6.655738094879564, 0.3665565060573631, 7.441773442627868, 0.319338644361316, 4.592207675647402e-06, 2, 1, 1.0 * mdp.action_std]
        else
            ps_pomcpow = [70.89349510683897, 5.234934973984675, 0.6711426346729477, 0.3721632872359978, 0.47654280344596933]
            ps_pftdpw = [135.0734110786185, 3.7856012796559453, 0.7103696024678953, 0.5857150350834478, 0.723483805464025]
            ps_agmcts = [173.42526232816897, 1.279248523870609, 0.5358669293276164, 6.393780537818737, 0.25516369283436635, 5.8412093448239695e-05, 2, 1, 1.0 * mdp.action_std]
        end

        action_optim_iters = 3
        update_actions_imp_ratio_add_threshold = 1.0
        update_actions_imp_ratio_delete_threshold = 0.5
        use_mc_immediate_reward = false
        sample_state_reward_gradient = true
        grad_weighted_by_visits = false
        choose_random_obs = true
        action_update_rule = ActionUpdateMinChildrenEveryKMaxADist(ps_agmcts[7], ps_agmcts[8], ps_agmcts[9])
    elseif !simple_true_ode_false && !mdp_true_pomdp_false
        if !voo
            ps_pomcpow = [101.513714168354, 9.942343584484084, 0.18220327350973564, 8.563172327068136, 0.8395219840102979]
            ps_pftdpw = [162.85851510992256, 10.313625885071149, 0.55539946796431, 0.8537230602475799, 0.5344691382180428]
            ps_agmcts = [131.40832609327452, 5.551124741173714, 0.30270003047663735, 9.879297887121709, 0.578291322022643, 8.072081764786078e-06, 2, 1, 1.0 * mdp.action_std]
        else
            ps_pomcpow = [70.12611523903757, 7.4376868962756815, 0.3631236093960938, 6.813571518938267, 0.7249984981381423]
            ps_pftdpw = [5.235526668477584e-07, 4.8010320072444665, 0.49025331412616624, 8.149513474157443, 0.3153114162158495]
            ps_agmcts = [136.44703410621412, 1.9510892478298043, 0.5465904615311035, 13.305318153373417, 0.41833111230918074, 1.452197521662336e-05, 2, 1, 1.0 * mdp.action_std]
        end

        action_optim_iters = 2
        update_actions_imp_ratio_add_threshold = 0.99
        update_actions_imp_ratio_delete_threshold = 1e-2
        use_mc_immediate_reward = true
        sample_state_reward_gradient = true
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
        solver_gen=mountaincar_baseline_solver,
        solver_name="ThresholdVel_$(baseline_solver_threshold_vel)"
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
        linearize_weight_update=true,
        grad_weighted_by_visits=grad_weighted_by_visits,
        action_update_rule=action_update_rule,
        update_actions_imp_ratio_add_threshold=update_actions_imp_ratio_add_threshold,
        update_actions_imp_ratio_delete_threshold=update_actions_imp_ratio_delete_threshold,
        optimize_before_update=true,
        choose_random_obs=choose_random_obs,
        value_updater=ValueUpdaterSNMISMC()
    )

    mountaincar_default_action = (i) -> BasicPOMCP.ReportWhenUsed(0.0)
    # mountaincar_default_action = (i) -> BasicPOMCP.ExceptionRethrow()

    return pomdp, a_type, sim_params, baseline_solver_params, ps_pomcpow, pomcpow_params, ps_pftdpw, pftdpw_params, ps_agmcts, agmcts_params, lr_to_opt_func, rollout_k_samples, mountaincar_ev, mountaincar_na_rand, mountaincar_na_voo, mountaincar_default_action
end

function simulation_probmountaincar(mdp_true_pomdp_false=false, simple_true_ode_false=true; solver="ag-dpw")
    ### Parameter objects initialization
    voo = solver_uses_voo(solver)
    pomdp, a_type, sim_params, baseline_solver_params, ps_pomcpow, pomcpow_params, ps_pftdpw, pftdpw_params, ps_agmcts, agmcts_params, lr_to_opt_func, rollout_k_samples, mountaincar_ev, mountaincar_na_rand, mountaincar_na_voo, mountaincar_default_action = scenario_parameters(mdp_true_pomdp_false, simple_true_ode_false, voo)
    p = mdp_true_pomdp_false ? pomdp.mdp : pomdp

    pftdpw_params.tree_in_info = true
    agmcts_params.tree_in_info = true

    sim_params_warmup = deepcopy(sim_params)
    sim_params_warmup.num_sims = use_distributed ? num_workers : 1
    sim_params.num_sims = use_distributed ? 20 : 2

    prob_name = "ProbMountainCar" * (mdp_true_pomdp_false ? "MDP" : "POMDP") * (simple_true_ode_false ? "Simple" : "ODE")
    out_name_init = "gifs/ProbMountainCar/$(prob_name)/$(prob_name)"
    out_filename = (metadata) -> out_name_init * "$(metadata[:k])_$(metadata[:i])"
    # gif_process = make_gif_process(out_filename, 1, true)
    fps = simple_true_ode_false ? 25 : 5
    # process = make_gif_process(out_filename, fps, true)
    # process = make_gif_process_xva_plot(out_filename, fps, true)
    # process = D == 2 ? make_pdf_process(out_filename, 1, true) : make_print_history_process()
    process = make_print_history_process()

    @info "warmup for compilation"
    sims_baseline = gen_sims_solver(m=p,
        solver_params=baseline_solver_params,
        sim_params=sim_params_warmup
    )
    if !mdp_true_pomdp_false && solver_is_pomcpow(solver)
        @info "Running simulations ProbMountainCar $(canonical_solver_name(solver; mdp=mdp_true_pomdp_false))"
        sims_pomcpow = gen_sims(m=p,
            solver_params=pomcpow_params,
            sim_params=sim_params,
            estimate_value=mountaincar_ev,
            next_action=voo ? mountaincar_na_voo : mountaincar_na_rand,
            default_action=mountaincar_default_action
        )
        gather_sim_results(sims_pomcpow, use_distributed, process; print_results=true)
    end

    if solver_is_dpw(solver)
        @info "Running simulations ProbMountainCar $(canonical_solver_name(solver; mdp=mdp_true_pomdp_false))"
        sims_pftdpw = gen_sims(m=p,
            solver_params=pftdpw_params,
            sim_params=sim_params,
            estimate_value=mountaincar_ev,
            next_action=voo ? mountaincar_na_voo : mountaincar_na_rand,
            default_action=mountaincar_default_action
        )
        gather_sim_results(sims_pftdpw, use_distributed, process; print_results=true)
    end

    if solver_is_agmcts(solver)
        @info "Running simulations ProbMountainCar $(canonical_solver_name(solver; mdp=mdp_true_pomdp_false))"
        sims_agmcts = gen_sims(m=p,
            solver_params=agmcts_params,
            sim_params=sim_params,
            estimate_value=mountaincar_ev,
            next_action=voo ? mountaincar_na_voo : mountaincar_na_rand,
            default_action=mountaincar_default_action
        )
        gather_sim_results(sims_agmcts, use_distributed, process; print_results=true)
    end

    return
end

function ce_opt_mountaincar(; mdp_true_pomdp_false=false, simple_true_ode_false=true, solver="ag-dpw", optim_option::Int=3, test_mode=false)
    opt_iters = 50
    num_param_samples = 150
    num_elite_samples = 30
    smooth_alpha = [0.8, 0.5]
    score_norm_mountaincar = (x, se) -> (x - se) / 60  # Linear mapping from [-60, 60] to [-1,1]

    voo = solver_uses_voo(solver)
    pomdp, a_type, sim_params, baseline_solver_params, ps_pomcpow, pomcpow_params, ps_pftdpw, pftdpw_params, ps_agmcts, agmcts_params, lr_to_opt_func, rollout_k_samples, mountaincar_ev, mountaincar_na_rand, mountaincar_na_voo, mountaincar_default_action = scenario_parameters(mdp_true_pomdp_false, simple_true_ode_false, voo, optim_option)

    sim_params.num_sims = use_distributed ? 40 : 2

    if test_mode
        opt_iters = 2
        num_param_samples = 12
        num_elite_samples = 10
        sim_params.num_sims = 2
    end

    prob = mdp_true_pomdp_false ? pomdp.mdp : pomdp
    prob_name = "ProbMountainCar" * (mdp_true_pomdp_false ? "MDP" : "POMDP") * (simple_true_ode_false ? "Simple" : "ODE")

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
    init_cov_agmcts = diagm([4.0, 3.0, 4.0, 3.0, 4.0, 3.0] .^ 2)
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
            prob_ev=mountaincar_ev,
            prob_na=voo ? mountaincar_na_voo : mountaincar_na_rand,
            prob_da=mountaincar_default_action,
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
            score_norm=score_norm_mountaincar,
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
            prob_ev=mountaincar_ev,
            prob_na=voo ? mountaincar_na_voo : mountaincar_na_rand,
            prob_da=mountaincar_default_action,
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
            score_norm=score_norm_mountaincar,
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
            prob_ev=mountaincar_ev,
            prob_na=voo ? mountaincar_na_voo : mountaincar_na_rand,
            prob_da=mountaincar_default_action,
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
            score_norm=score_norm_mountaincar,
            params_empty_csv=params_empty_csv_agmcts,
            smooth_alpha=smooth_alpha,
            # term_eps::Float64=0.1,
            rejection_function=rejection_function_agmcts,
            # General
            parallelize=use_distributed,
        )
    end
end

function max_time_mountaincar(; mdp_true_pomdp_false=false, simple_true_ode_false=true, solver="ag-dpw", optim_option::Int=3, test_mode=false)
    voo = solver_uses_voo(solver)
    pomdp, a_type, sim_params, baseline_solver_params, ps_pomcpow, pomcpow_params, ps_pftdpw, pftdpw_params, ps_agmcts, agmcts_params, lr_to_opt_func, rollout_k_samples, mountaincar_ev, mountaincar_na_rand, mountaincar_na_voo, mountaincar_default_action = scenario_parameters(mdp_true_pomdp_false, simple_true_ode_false, voo, optim_option)

    max_time_list = [30.0]
    max_query = 500
    max_query_list = map(x -> round(Int, max_query * x), 10.0 .^ (-1.0:0.25:0.0))

    max_query_list_pomcpow = [round(Int, nq * sqrt(pftdpw_params.num_particles_planner)) for nq in max_query_list]
    sim_params.num_sims = test_mode ? 10 : 1000

    prob = mdp_true_pomdp_false ? pomdp.mdp : pomdp
    prob_name = "ProbMountainCar" * (mdp_true_pomdp_false ? "MDP" : "POMDP") * (simple_true_ode_false ? "Simple" : "ODE")

    if !mdp_true_pomdp_false && solver_is_pomcpow(solver)
        compare_by_max_time_query(; m=prob,
            solvers_params=[[pomcpow_params, sim_params, mountaincar_ev, voo ? mountaincar_na_voo : mountaincar_na_rand, mountaincar_default_action]],
            prob_name=prob_name,
            parallelize=use_distributed,
            max_time_list=max_time_list,
            max_query_list=max_query_list_pomcpow
        )
    end

    if solver_is_dpw(solver)
        compare_by_max_time_query(; m=prob,
            solvers_params=[[pftdpw_params, sim_params, mountaincar_ev, voo ? mountaincar_na_voo : mountaincar_na_rand, mountaincar_default_action]],
            prob_name=prob_name,
            parallelize=use_distributed,
            max_time_list=max_time_list,
            max_query_list=max_query_list
        )
    end

    if solver_is_agmcts(solver)
        compare_by_max_time_query(; m=prob,
            solvers_params=[[agmcts_params, sim_params, mountaincar_ev, voo ? mountaincar_na_voo : mountaincar_na_rand, mountaincar_default_action]],
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
        simulation_probmountaincar(args["mdp"],
            args["simple"];
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
        ce_opt_mountaincar(
            mdp_true_pomdp_false=args["mdp"],
            simple_true_ode_false=args["simple"],
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
        max_time_mountaincar(
            mdp_true_pomdp_false=args["mdp"],
            simple_true_ode_false=args["simple"],
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
