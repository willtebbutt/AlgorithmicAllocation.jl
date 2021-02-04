a = [1:10;]
Base.unsafe_wrap(Array, pointer(a) + 1 * sizeof(eltype(a)), 5)
C.@preserve a
