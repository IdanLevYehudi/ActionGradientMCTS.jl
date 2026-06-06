macro forward_fields(type_name, inner_field, field_names)
    quote
        # Define getproperty method for the specified type
        function Base.getproperty(obj::$(esc(type_name)), name::Symbol)
            # Check if the requested field name is one of the specified fields to forward
            if name in ($(field_names))
                return getfield(getfield(obj, $(esc(inner_field))), name)
            else
                return getfield(obj, name)
            end
        end
    end
end

abstract type AbstractActionOptimizer end

struct GradAscentActionOptimizer <: AbstractActionOptimizer
    step_size::Float64
end

struct FluxOptimizer{A} <: AbstractActionOptimizer
    flux_opt::Flux.Optimise.AbstractOptimiser
    optimizers::Dict{Int,Flux.Optimise.AbstractOptimiser}
    action_accumulators::Dict{Int,A}

    function FluxOptimizer{A}(opt::Opt) where {A,Opt<:Flux.Optimise.AbstractOptimiser}
        new(opt,
            Dict{Int,Flux.Optimise.AbstractOptimiser}(),
            Dict{Int,A}()
        )
    end
end

is_iterable(obj) = Base.IteratorSize(typeof(obj)) != Base.HasShape{0}()

function optimize_node(fopt::FluxOptimizer{A}, index::Int, params, grad) where {A}
    if !haskey(fopt.optimizers, index)
        fopt.optimizers[index] = deepcopy(fopt.flux_opt)
        fopt.action_accumulators[index] = deepcopy(params)
    end

    params_copy = fopt.action_accumulators[index]
    grad_copy = grad

    iterable = true
    if !is_iterable(params_copy)  # Meaning object is not iterable
        iterable = false
        params_copy = [params_copy]
        grad_copy = [grad_copy]
    end

    params_copy = params_copy isa SArray ? MArray(params_copy) : params_copy
    grad_copy = grad_copy isa SArray ? MArray(grad_copy) : grad_copy

    # Calculate the update
    delta = zero(params_copy)
    try
        delta = Flux.Optimise.apply!(fopt.optimizers[index], deepcopy(params_copy), deepcopy(grad_copy))
    catch ex
        if ex isa InterruptException
            throw(ex)
        end
        @warn "Caught exception in Flux optimizer" maxlog = 3
        @warn ex maxlog = 3
    end

    if !iterable
        new_action = params_copy[1] + delta[1]  # Plus sign instead of minus for gradient ascent
    else
        new_action = params_copy .+ delta  # Plus sign instead of minus for gradient ascent
    end
    new_action = typeof(params)(new_action)

    fopt.action_accumulators[index] = new_action

    return new_action
end
