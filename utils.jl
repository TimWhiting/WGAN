module utils

"""
Returns a function that takes a single argument `dims` and returns
uniformly sampled random values matching shape `dims` with mean `μ`
and upper and lower bound of `(μ - bound, μ + bound)`.
"""
randuFn(μ::Number, bound::Number) = (dims...) -> (rand(dims...) .* 2 .- 1) .* bound .+ μ

end # module utils