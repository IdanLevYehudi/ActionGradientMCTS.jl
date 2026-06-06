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

function tests_collablightdark2d()
    D = 2
    K = 3
    pomdp = CollabLightDarkPOMDP{D,K}()
    mdp = UnderlyingMDP(pomdp)
    s = rand(initialstate(pomdp))

    straight_to_goal_solver = CLDStraightToGoalSolver(rng=Xoshiro(1235 + 30_000))
    straight_to_goal_policy = solve(straight_to_goal_solver, pomdp)
    distances_j = [norm(d_slice(s, D, i) - mdp.goal_position) for i in 1:K]
    sp = deepcopy(s)
    @assert !isterminal(pomdp, sp)
    for j in 1:5
        println(distances_j)
        if isterminal(pomdp, sp)
            break
        end

        a_policy = action(straight_to_goal_policy, s)
        @assert a_policy isa SVector{D * K,Float64}

        sp = rand(transition(pomdp, s, a_policy))
        distances_j_1 = [norm(d_slice(sp, D, i) - pomdp.mdp.goal_position) for i in 1:K]
        @assert all(distances_j_1 .<= distances_j)
        @assert sp isa SVector{D * K,Float64}

        o_dist = observation(pomdp, sp)
        o = rand(o_dist)
        @assert o isa SVector{D * K,Float64}
        @assert 0 < pdf(o_dist, o) < 1e6
        sp_diff = CollabLightDark.obs_pos_diff(pomdp, sp)
        println(o_dist.dist.Σ.diag)
        for i in 1:K
            d = norm(d_slice(sp_diff, D, i))
            sigma_slice = d_slice(o_dist.dist.Σ.diag, D, i)
            @assert all(sigma_slice .<= pomdp.max_obs_std .^ 2)
            std_calc = min(pomdp.max_obs_std, pomdp.meas_std_by_dist_k * d^pomdp.meas_std_by_dist_alpha)
            @assert(all(sigma_slice .== std_calc^2))
        end

        distances_j = distances_j_1
        s = sp
    end
    @assert isterminal(pomdp, sp) == true
end

function tests_gradients_collablightdark()
    D = 3
    K = 1
    pomdp = CollabLightDarkPOMDP{D,K}()
    mdp = UnderlyingMDP(pomdp)
    s = rand(initialstate(pomdp))
    a = rand(actions(pomdp, s))
    println("Sampled s: $s")
    println("Sampled a: $a")

    n = 100
    rng_node_updater = Xoshiro(1234 + 30_000)
    node_updater = BootstrapFilter(deepcopy(pomdp), n, rng_node_updater)
    belief_mdp = GenerativeBeliefPropMDP(deepcopy(pomdp), node_updater)
    b = ParticleCollection([rand(initialstate(pomdp)) for i in 1:n])
    println("Checking known displacement")
    for dx in [0:0.01:0.1;]
        println("dx: $dx")
        bp = ParticleCollection([(p + a) .+ dx for p in b.particles])
        println("transition_log_likelihood for ordered belief: ", transition_log_likelihood(belief_mdp, b, a, bp), " ", exp(transition_log_likelihood(belief_mdp, b, a, bp)))
        println("transition_log_likelihood2 for unordered belief: ", transition_log_likelihood2(belief_mdp, b, a, bp), " ", exp(transition_log_likelihood2(belief_mdp, b, a, bp)))
    end
    println("Checking random displacement")
    for sigma in [0.01:0.01:0.2;]
        println("sigma: $sigma")
        bp = typeof(b)([(p + a) .+ sigma .* typeof(a)(randn(length(a))) for p in b.particles])
        println("transition_log_likelihood for ordered belief: ", transition_log_likelihood(belief_mdp, b, a, bp), " ", exp(transition_log_likelihood(belief_mdp, b, a, bp)))
        println("transition_log_likelihood2 for unordered belief: ", transition_log_likelihood2(belief_mdp, b, a, bp), " ", exp(transition_log_likelihood2(belief_mdp, b, a, bp)))
    end
end

function timing_tests_collablightdark2d()
    D = 2
    K = 2
    pomdp = CollabLightDarkPOMDP{D,K}()
    println("Benchmarking state functions")
    s = rand(initialstate(pomdp))
    a = rand(actions(pomdp, s))
    sp = rand(transition(pomdp, s, a))
    o = rand(observation(pomdp, sp))
    println("Sampled s: $s")
    println("Sampled a: $a")
    println("Sampled sp: $sp")
    println("Sampled o: $o")

    println("Benchmarking obs_pos_diff")
    @btime $obs_pos_diff($pomdp, $sp)

    println("Benchmarking observation generation")
    @btime $rand($observation($pomdp, $sp))

    println("Benchmarking observation pdf")
    @btime $pdf($observation($pomdp, $sp), $o)

    println("Benchmarking state gradlogpdf")
    @btime $gradlogpdf($transition($pomdp, $s, $a), $sp)

    println("Benchmarking state reward")
    @btime $reward($pomdp, $s, $a, $sp)

    println("Benchmarking state grad_reward")
    @btime $grad_reward($pomdp, $s, $a, $sp)

    println("Benchmarking transition sampling")
    @btime $rand($transition($pomdp, $s, $a))

    println("Benchmarking state transition_log_likelihood")
    @btime $transition_log_likelihood($pomdp, $s, $a, $sp)

    println("Benchmarking state grad_log_transition")
    @btime $grad_log_transition($pomdp, $s, $a, $sp)

    n = 100
    println("Benchmarking belief functions for $n particles")
    rng_node_updater = Xoshiro(1234 + 30_000)
    node_updater = BootstrapFilter(deepcopy(pomdp), n, rng_node_updater)
    belief_mdp = GenerativeBeliefPropMDP(deepcopy(pomdp), node_updater)
    b = ParticleCollection([rand(initialstate(pomdp)) for i in 1:n])
    dx = 0.05
    bp = ParticleCollection([(p + a) .+ dx for p in b.particles])

    println("Benchmarking belief transition_log_likelihood")
    @btime $transition_log_likelihood($belief_mdp, $b, $a, $bp)

    println("Benchmarking belief grad_log_transition")
    @btime $grad_log_transition($belief_mdp, $b, $a, $bp)

    println("Benchmarking belief grad_reward")
    @btime $grad_reward($belief_mdp, $b, $a, $bp)
end

function cld_create_ev_and_na(D, K, pomdp, sigma, rollout_k_samples)
    cld_ev = (i) -> MeanStateRollout(CLDStraightToGoalSolver(rng=Xoshiro(1235 + 30_000 * i), sigma=sigma), rollout_k_samples)

    randon_action_generator = (i) -> RandomActionGenerator(Xoshiro(1242 + 30_000 * i))

    # VOO parameters
    voo_p = 0.85
    voo_exp_sampler = Distributions.Uniform()
    voo_sigs = [0.05 for _ in 1:(D*K)]
    voo_sig = diagm(voo_sigs)
    voo_voronoi_sampler = MvNormal(voo_sig)
    voo_accept_radius_sq = 0.01^2 * D * K
    voo_rng = (i) -> Xoshiro(4321 + 30_000 * i)
    voo_action_generator = (i) -> VOOActionGenerator(voo_p, voo_exp_sampler, voo_voronoi_sampler, voo_rng(i), voo_accept_radius_sq, true, 20)

    cld_na_gen = (gen) -> ((i) -> PolicyFirstGen(CLDStraightToGoalPolicy(prob=UnderlyingMDP(deepcopy(pomdp)), rng=Xoshiro(1235 + 30_000 * i), sigma=sigma), gen(i)))
    return cld_ev, cld_na_gen(randon_action_generator), cld_na_gen(voo_action_generator)
end

function scenario_parameters(D, K, voo=false, optim_option::Int=3)
    start_pos = SVector{D,Float64}(zeros(D))
    goal_pos = SVector{D,Float64}(vcat(zeros(D - 1), 2.5))
    beacon_pos = ((start_pos + goal_pos) / 2) + SVector{D,Float64}(vcat(2.5, zeros(D - 1)))
    obstacles = Vector{SVector{D,Float64}}()
    obstacles_radii = Vector{Float64}()
    pomdp = CollabLightDarkPOMDP{D,K}(;
        mdp=CollabLightDarkMDP{D,K}(
            action_max_radius=1.5,
            goal_position=goal_pos,
            goal_tolerance=0.2,
            obstacles=obstacles,
            obstacles_radii=obstacles_radii,
            agent_transition_std=0.03,
            b0_radius=0.3,
            discount=0.99,
            reward_penalty_goal_distsq=-0.02,
            reward_success=10.0,
            reward_penalty_obstacles=-5.0,
            max_action_prob=0.0,
        ),
        beacon_pos=beacon_pos,
        meas_std_by_dist_k=0.01,
        meas_std_by_dist_alpha=8.0,
        max_obs_std=15.0,
    )
    a_type = typeof(rand(actions(pomdp)))

    sim_max_steps = 8
    num_particles_sim = [128, 256, 512, 1024, 2048, 4096, 4096, 8192][D*K]

    ### Solver params
    max_tree_depth = 8
    solver_timeout = 30.0
    max_query = 1000
    num_particles_planner = [10, 64, 128, 256, 200, 250, 300, 350][D*K]
    max_query_pomcpow = round(Int, max_query * sqrt(num_particles_planner))

    action_optim_iters = [1, 3, 3, 3, 5, 5, 5, 5][D*K]
    grad_mc_k_s = 10
    grad_mc_k_obs = 2
    rollout_k_samples = [1, 4, 4, 4, 5, 5, 5, 5][D*K]  # Unused
    grad_mc_k_particles = [1, 4, 4, 4, 10, 10, 10, 10][D*K]

    update_actions_imp_ratio_add_threshold = 0.9
    update_actions_imp_ratio_delete_threshold = 1e-8
    adam_beta = (0.9, 0.999)

    if D == 2 && K == 1
        gradient_clip = 500.0
    elseif D == 3 && K == 1
        gradient_clip = 250.0
    elseif D == 4 && K == 1
        gradient_clip = 187.0
    elseif D == 2 && K == 2
        gradient_clip = 382.0
    end

    exp_decay_params = [1.0, 0.999, 1, 0.1, 0]

    if optim_option == 1
        lr_to_opt_func = (lr) -> FluxOptimizer{a_type}(Optimiser(Momentum(lr, adam_beta[1]), ExpDecay(exp_decay_params...)))
    elseif optim_option == 2
        lr_to_opt_func = (lr) -> FluxOptimizer{a_type}(Optimiser(ClipNorm(gradient_clip), Momentum(lr, adam_beta[1]), ExpDecay(exp_decay_params...)))
    elseif optim_option == 3
        lr_to_opt_func = (lr) -> FluxOptimizer{a_type}(Optimiser(Adam(lr, adam_beta), ExpDecay(exp_decay_params...)))
    elseif optim_option == 4
        lr_to_opt_func = (lr) -> FluxOptimizer{a_type}(Optimiser(ClipNorm(gradient_clip), Adam(lr, adam_beta), ExpDecay(exp_decay_params...)))
    end

    if D == 2 && K == 1
        if !voo
            ps_pomcpow = [0.8644280765017149, 0.45763896993938785, 0.7677196548522844, 0.1576741225533416, 0.24643935545200382]
            ps_pftdpw = [1.010551075092455, 7.675341443334524, 0.5153265944577876, 8.901233212904298, 0.30365697323026786]
            ps_agmcts = [1.9630787464912196, 5.697086924582306, 0.4727920467092924, 8.052361875479292, 0.7693105325960976, 5.299978373546704e-08, 2, 1, 5e-2 * pomdp.mdp.agent_transition_std]
        else
            ps_pomcpow = [1.4511796816162321, 0.3635175567917663, 0.7454480330210095, 0.09628501282744448, 0.2999517771770834]
            ps_pftdpw = [1.0206384755585345, 7.160758913932819, 0.48179503658817724, 9.028968717043224, 0.34704732177344566]
            ps_agmcts = [3.1649861007355033, 4.291478419362525, 0.4810085549749057, 4.50852434460785, 0.6683817748851235, 4.283772552157376e-07, 2, 1, 5e-2 * pomdp.mdp.agent_transition_std]
        end
    end

    if D == 3 && K == 1
        if !voo
            ps_pomcpow = [1.0401765791895432, 0.850890901832541, 0.4880851541526753, 0.40418345624422825, 0.3265991675572641]
            ps_pftdpw = [1.7030561935733175, 11.76232815837249, 0.2646854182906825, 7.485562722872231, 0.3303716388857176]
            ps_agmcts = [2.516042127258674, 5.705196985747078, 0.4391710352038719, 6.113184073444049, 0.6261666233874854, 3.760373011642239e-08, 2, 1, 5e-2 * pomdp.mdp.agent_transition_std]
        else
            ps_pomcpow = [1.2842316509412566, 0.45475573458895513, 0.7674706029283438, 0.15771943177756942, 0.2838175955005755]
            ps_pftdpw = [1.2808333620832208, 7.289712045415534, 0.5364352777274091, 8.575326789091184, 0.6329504527075918]
            ps_agmcts = [1.6009002715223657, 4.5728838779633625, 0.6241958070258606, 8.933953096375422, 0.22492820907934127, 2.873268082766309e-07, 2, 1, 5e-2 * pomdp.mdp.agent_transition_std]
        end
    end

    if D == 4 && K == 1
        if !voo
            ps_pomcpow = [1.1763524658266933, 0.7901001882642544, 0.4708875181136456, 0.274738270475642, 0.4149292061568313]
            ps_pftdpw = [1.453962588163547, 10.849845732365353, 0.2863048590775734, 7.001923471257392, 0.13925705414876824]
            ps_agmcts = [3.5868382080405494, 7.82245138682696, 0.22447558243490479, 0.9617798580737597, 0.2423217649095438, 4.214127088528494e-08, 2, 1, 5e-2 * pomdp.mdp.agent_transition_std]
        else
            ps_pomcpow = [1.5472435878651425, 0.31247848672984435, 0.8151769443834023, 0.14382943632524992, 0.1439880360614939]
            ps_pftdpw = [1.1198469152871693, 8.388496937259834, 0.45846084849616714, 6.7413073483685, 0.6205888555548617]
            ps_agmcts = [0.7000585336419516, 5.124039738834746, 0.617151144241942, 6.205420607866507, 0.32715545018172487, 2.7719086802829854e-08, 2, 1, 5e-2 * pomdp.mdp.agent_transition_std]
        end
    end

    if D == 2 && K == 2
        if !voo
            ps_pomcpow = [4.124357735559094, 4.064172095943853, 0.2977746766389674, 3.9652440704377296, 0.36916247694351073]
            ps_pftdpw = [3.7234553851084233, 9.635759008842824, 0.07771671494984067, 2.885899005160133, 0.40380489145550746]
            ps_agmcts = [9.3276775063348, 0.14205857378529738, 0.2610973853697225, 10.808770613295001, 0.49526757893505857, 0.012207233599308134, 2, 1, 5e-2 * pomdp.mdp.agent_transition_std]
        else
            ps_pomcpow = [1.758463626158285, 4.798062331670242, 0.5205364837361538, 1.703028608200049, 0.5178338070234411]
            ps_pftdpw = [0.9271462411148794, 9.240853118846468, 0.4909691604646041, 8.135076978695135, 0.4349979477332005]
            ps_agmcts = [0.0067041248058604735, 0.0007516501696948247, 0.9998163825391752, 18.986481746546804, 0.9984062503502861, 0.899465307825196, 2, 1, 5e-2 * pomdp.mdp.agent_transition_std]
        end
    end

    sim_params = SimParams(
        sim_max_steps=sim_max_steps,
        num_sims=1,
        num_particles_sim=num_particles_sim
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
        use_mc_immediate_reward=true,
        sample_state_reward_gradient=true,
        linearize_weight_update=true,
        grad_weighted_by_visits=true,
        action_update_rule=ActionUpdateMinChildrenEveryKMaxADist(ps_agmcts[7], ps_agmcts[8], ps_agmcts[9]),
        update_actions_imp_ratio_add_threshold=update_actions_imp_ratio_add_threshold,
        update_actions_imp_ratio_delete_threshold=update_actions_imp_ratio_delete_threshold,
        optimize_before_update=true,
        choose_random_obs=true,
        value_updater=ValueUpdaterSNMISMC()
    )

    policy_sigma = 0.1
    cld_ev, cld_na_rand, cld_na_voo = cld_create_ev_and_na(D, K, pomdp, policy_sigma, rollout_k_samples)
    cld_default_action = (i) -> BasicPOMCP.ReportWhenUsed(zero(SVector{D * K,Float64}))
    # cld_default_action = (i) -> BasicPOMCP.ExceptionRethrow()

    return pomdp, a_type, sim_params, ps_pomcpow, pomcpow_params, ps_pftdpw, pftdpw_params, ps_agmcts, agmcts_params, lr_to_opt_func, policy_sigma, rollout_k_samples, cld_ev, cld_na_rand, cld_na_voo, cld_default_action
end

function simulation_collablightdark(D::Int, K::Int, optim_option::Int=3; solver="ag-dpw")
    ### Parameter objects initialization
    voo = solver_uses_voo(solver)
    pomdp, a_type, sim_params, ps_pomcpow, pomcpow_params, ps_pftdpw, pftdpw_params, ps_agmcts, agmcts_params, lr_to_opt_func, policy_sigma, rollout_k_samples, cld_ev, cld_na_rand, cld_na_voo, cld_default_action = scenario_parameters(D, K, voo, optim_option)

    agmcts_params.tree_in_info = true

    sim_params_warmup = deepcopy(sim_params)
    sim_params_warmup.num_sims = use_distributed ? num_workers : 1
    sim_params.num_sims = use_distributed ? 12 : 8

    out_name_init = "gifs/CollabLightDark_$(D)_$(K)_"
    out_filename = (metadata) -> out_name_init * "$(metadata[:k])_$(metadata[:i])"
    # gif_process = make_gif_process(out_filename, 1, true)
    # process = D == 2 ? make_gif_process(out_filename, 1, true) : make_print_history_process()
    process = make_print_history_process()

    @info "warmup for compilation"
    if solver_is_pomcpow(solver)
        @info "Running simulations CollabLightDark{$D,$K} $(canonical_solver_name(solver; mdp=false))"
        sims_pomcpow = gen_sims(m=pomdp,
            solver_params=pomcpow_params,
            sim_params=sim_params,
            estimate_value=cld_ev,
            next_action=voo ? cld_na_voo : cld_na_rand,
            default_action=cld_default_action
        )
        gather_sim_results(sims_pomcpow, use_distributed, process; print_results=true)
    end

    if solver_is_dpw(solver)
        @info "Running simulations CollabLightDark{$D,$K} $(canonical_solver_name(solver; mdp=false))"
        sims_pftdpw = gen_sims(m=pomdp,
            solver_params=pftdpw_params,
            sim_params=sim_params,
            estimate_value=cld_ev,
            next_action=voo ? cld_na_voo : cld_na_rand,
            default_action=cld_default_action
        )
        gather_sim_results(sims_pftdpw, use_distributed, process; print_results=true)
    end

    if solver_is_agmcts(solver)
        @info "Running simulations CollabLightDark{$D,$K} $(canonical_solver_name(solver; mdp=false))"
        sims_agmcts = gen_sims(m=pomdp,
            solver_params=agmcts_params,
            sim_params=sim_params,
            estimate_value=cld_ev,
            next_action=voo ? cld_na_voo : cld_na_rand,
            default_action=cld_default_action
        )
        gather_sim_results(sims_agmcts, use_distributed, process; print_results=true)
    end

    return
end

function main_tests()
    tests_collablightdark2d()
    tests_gradients_collablightdark()
    timing_tests_collablightdark2d()
end

function ce_opt_collablightdark(; D::Int, K::Int, solver="ag-dpw", optim_option::Int=3, test_mode=false)
    opt_iters = 50
    num_param_samples = 150
    num_elite_samples = 30
    smooth_alpha = [0.8, 0.5]
    score_norm_cld = (x, se) -> (x - se - 5) / 5  # Maps [0, 10] to [-1, 1]

    voo = solver_uses_voo(solver)
    pomdp, a_type, sim_params, ps_pomcpow, pomcpow_params, ps_pftdpw, pftdpw_params, ps_agmcts, agmcts_params, lr_to_opt_func, policy_sigma, rollout_k_samples, cld_ev, cld_na_rand, cld_na_voo, cld_default_action = scenario_parameters(D, K, voo, optim_option)

    sim_params.num_sims = use_distributed ? 40 : 2

    if test_mode
        opt_iters = 2
        num_param_samples = 12
        num_elite_samples = 10
        sim_params.num_sims = 2
    end

    prob_name_pomdp = "CLD_$(D)_$(K)_psigma_$(policy_sigma)"

    ### POMCPOW params
    init_params_container_pomcpow = pomcpow_params
    params_empty_csv_pomcpow = DataFrame(ucb_c=Float64[], k_act=Float64[], alpha_act=Float64[], k_obs=Float64[], alpha_obs=Float64[])
    params_log_func_pomcpow = (p) -> p
    params_exp_func_pomcpow = (p) -> p
    init_mean_pomcpow = ps_pomcpow
    init_logmean_pomcpow = params_log_func_pomcpow(init_mean_pomcpow)
    init_cov_pomcpow = diagm([4.0, 3.0, 0.5, 3.0, 0.5] .^ 2)
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
    params_log_func_agmcts = (p) -> [p[1], p[2], p[3] * 10.0, p[4], p[5] * 10.0, log(p[6])]
    params_exp_func_agmcts = (p) -> [p[1], p[2], p[3] / 10.0, p[4], p[5] / 10.0, exp(p[6])]
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
            prob=pomdp,
            prob_name=prob_name_pomdp,
            sim_params=sim_params,
            prob_ev=cld_ev,
            prob_na=voo ? cld_na_voo : cld_na_rand,
            prob_da=cld_default_action,
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
            score_norm=score_norm_cld,
            params_empty_csv=params_empty_csv_pftdpw,
            smooth_alpha=smooth_alpha,
            # term_eps::Float64=0.1,
            # General
            parallelize=use_distributed,
        )
    end

    ## POMCPOW
    if solver_is_pomcpow(solver)
        cross_entropy_optimization(;
            # Problem related parameters
            prob=pomdp,
            prob_name=prob_name_pomdp,
            sim_params=sim_params,
            prob_ev=cld_ev,
            prob_na=voo ? cld_na_voo : cld_na_rand,
            prob_da=cld_default_action,
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
            score_norm=score_norm_cld,
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
            prob=pomdp,
            prob_name=prob_name_pomdp,
            sim_params=sim_params,
            prob_ev=cld_ev,
            prob_na=voo ? cld_na_voo : cld_na_rand,
            prob_da=cld_default_action,
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
            score_norm=score_norm_cld,
            params_empty_csv=params_empty_csv_agmcts,
            smooth_alpha=smooth_alpha,
            # term_eps::Float64=0.1,
            rejection_function=rejection_function_agmcts,
            # General
            parallelize=use_distributed,
        )
    end

end

function max_time_collablightdark(; D::Int, K::Int, solver="ag-dpw", optim_option::Int=3, test_mode=false)
    voo = solver_uses_voo(solver)
    pomdp, a_type, sim_params, ps_pomcpow, pomcpow_params, ps_pftdpw, pftdpw_params, ps_agmcts, agmcts_params, lr_to_opt_func, policy_sigma, rollout_k_samples, cld_ev, cld_na_rand, cld_na_voo, cld_default_action = scenario_parameters(D, K, voo, optim_option)

    max_time_list = [30.0]
    max_query = 500
    max_query_list = map(x -> round(Int, max_query * x), 10.0 .^ (-1.0:0.25:0.0))

    max_query_list_pomcpow = [round(Int, nq * sqrt(pftdpw_params.num_particles_planner)) for nq in max_query_list]
    sim_params.num_sims = test_mode ? 10 : 1000

    mdp_kwargs = first(Base.kwarg_decl.(methods(CollabLightDarkMDP{D,K})))
    pomdp_kwargs = first(Base.kwarg_decl.(methods(CollabLightDarkPOMDP{D,K})))
    to_kwargs(s, kwargs) = (; (name => getfield(s, name) for name in fieldnames(typeof(s)) if name in kwargs)...)

    prob_name_pomdp = "CLD_$(D)_$(K)_psigma_$(policy_sigma)"

    if solver_is_pomcpow(solver)
        compare_by_max_time_query(; m=pomdp,
            solvers_params=[[pomcpow_params, sim_params, cld_ev, voo ? cld_na_voo : cld_na_rand, cld_na_voo, cld_default_action]],
            prob_name=prob_name_pomdp,
            parallelize=use_distributed,
            max_time_list=max_time_list,
            max_query_list=max_query_list_pomcpow
        )
    end

    if solver_is_dpw(solver)
        compare_by_max_time_query(; m=pomdp,
            solvers_params=[[pftdpw_params, sim_params, cld_ev, voo ? cld_na_voo : cld_na_rand, cld_na_voo, cld_default_action]],
            prob_name=prob_name_pomdp,
            parallelize=use_distributed,
            max_time_list=max_time_list,
            max_query_list=max_query_list
        )
    end

    if solver_is_agmcts(solver)
        compare_by_max_time_query(; m=pomdp,
            solvers_params=[[agmcts_params, sim_params, cld_ev, voo ? cld_na_voo : cld_na_rand, cld_na_voo, cld_default_action]],
            prob_name=prob_name_pomdp,
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
        simulation_collablightdark(
            args["D"],
            args["K"];
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
        ce_opt_collablightdark(
            D=args["D"],
            K=args["K"],
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
        max_time_collablightdark(
            D=args["D"],
            K=args["K"],
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
