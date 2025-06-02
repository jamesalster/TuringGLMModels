
# Utility function for selecting draws and collapsing chains from a samples AxisArray
function process_draws(
    arr::AbstractArray;
    drop_warmup::Int=200, 
    n_draws::Int=-1, 
    collapse::Bool=true
)::NamedArray

    new_arr = arr[drop_warmup+1:end,:,:] #drop warmup
    new_arr = n_draws > 0 ? new_arr[1:n_draws,:,:] : new_arr #select draws
    !collapse && return new_arr
    # Collapse chains
    draws, params, chains = size(new_arr)
    collapsed_arr = reshape(permutedims(new_arr, [1,3,2]), draws*chains, params)
    return NamedArray(Array(collapsed_arr); names = (1:draws*chains, names(arr, 2)), dimnames = (:draw, :var))
end

# Mapslices for Named Array
function Base.mapslices(f::Function, NA::NamedArray; dims)
    out_arr = mapslices(f, Array(NA); dims)
    new_names = ntuple(i -> i ∈ dims ? collect(1:size(out_arr, i)) : names(NA)[i], ndims(NA))
    return NamedArray(out_arr; names = new_names, dimnames = Tuple(dimnames(NA)))
end

# Drop single dimensions where possible from a NamedArray
function drop_single_dims(NA::NamedArray)
    dims_to_drop = findall(==(1), size(NA))
    dims_to_keep = setdiff(1:ndims(NA), dims_to_drop)
    reduced_NA = dropdims(NA; dims = Tuple(dims_to_drop))
    new_names = ntuple(i -> names(NA, dims_to_keep[i]), length(dims_to_keep))
    new_dimnames = ntuple(i -> dimnames(NA, dims_to_keep[i]), length(dims_to_keep))
    return NamedArray(Array(reduced_NA); names = new_names, dimnames = new_dimnames)
end

# Helper function for display
function clean_prior_string(x)
    replace(x, r"\{.*\}" => "", "\n" => " ")
end

# Helper function to know TuringGLM's default links
function get_link(::Type{T}) where {T<:UnivariateDistribution}
    if T ∈ [Normal, TDist]
        return identity
    elseif T == Bernoulli
        return logit
    elseif T ∈ [Poisson, NegativeBinomial, TuringGLM.NegativeBinomial2]
        return log
    else
        @warn "Distribution $T unknown, assuming identity link"
        return identity
    end
end
