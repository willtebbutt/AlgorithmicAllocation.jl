a = [1:10;]
Base.unsafe_wrap(Array, pointer(a) + 1 * sizeof(eltype(a)), 5)
C.@preserve a


# The algorithm is present getting caught up on inconsequential blocks of code.
# We should certainly be able to elide computation if it's possible to prove that there
# won't be any iteration between blocks. Is this possible?
#
# It seems to be the case that for more-or-less any low-level function, you'll need to write
# a custom work-around rule to handle the lack of block-handling that this package presently
# provides. This is problematic.

foo(a, b) = a < b ? a : b

function bar(a, N)
    b = a
    for n in 1:N
        b += a
    end
    return b
end

@code_ir foo(5.0, 4.0)
@code_ir bar(5.0, 3)




# How to bake allocation analysis into preallocate?
function foo(x, y)
    a = bar(x, y)
    b = baz(a, x)
    return b
end

function preallocate(foo, x, y)
    a, mem_bar, pf_bar! = preallocate(bar, x, y)
    b, mem_baz, pf_baz! = preallocate(baz, a, x)
    pf_foo! = Preallocated{Tuple{typeof(foo), typeof(x), typeof(y)}}((pf_bar!, pf_baz!))
    return b, (mem_bar, mem_baz), pf_foo!
end

# Suppose that bar doesn't allocate, then `mem_bar` and `pf_bar!` should both be the
# (singleton) `DoesNotAllocate`. This can be used in `_preallocated`.

# If bot bar and baz allocate / we're not able to prove that they do not, then
# _preallocated should look something like this:
function _preallocated(p, mem, x, y)

    # Execute allocation-free bar.
    mem_bar = mem[1]
    pf_bar! = p.preallocated_functions[1]
    a = _preallocated(pf_bar!, mem_bar, x, y)

    # Execute allocation-free baz.
    mem_baz = mem[2]
    pf_baz! = p.preallocated_functions[2]
    b = _preallocated(pf_baz!, mem_baz, a, x)

    return b
end

# However, if `bar` doesn't allocate, we should obtain the following code:
function _preallocated(p, mem, x, y)

    # Execute bar as normal.
    a = bar(x, y)

    # Execute allocation-free baz.
    mem_baz = mem[2]
    pf_baz! = p.preallocated_functions[2]
    b = _preallocated(pf_baz!, mem_baz, a, x)

    return b
end

# This should be possible to achieve because the types of `p` and `mem` are available to
# use when generating the `_preallocated` function, meaning that we can condition on them
# to generate the code.

# It just remains to figure out how to generate the code needed to determine whether
# something will allocate or not, during the execution of `preallocate`. Something like this
# ought to work:

function preallocate(foo, x, y)

    # Construct pre-allocation data for bar.
    a, mem_bar, pf_bar! = if allocates(bar, x, y)
        preallocate(bar, x, y)
    else
        bar(x, y), DoesNotAllocate(), DoesNotAllocate()
    end

    # Construct pre-allocation data for bar.
    b, mem_baz, pf_baz! = if allocates(baz, a, x)
        preallocate(baz, a, x)
    else
        baz(a, x), DoesNotAllocate(), DoesNotAllocate()
    end

    # This is the same as before.
    pf_foo! = Preallocated{Tuple{typeof(foo), typeof(x), typeof(y)}}((pf_bar!, pf_baz!))
    return b, (mem_bar, mem_baz), pf_foo!
end

# This will work provided that we're able to implement `allocates`.
# In the base-cases, we clearly just define it manually.
# To derive it for a similar class of functions to those that we can derive automatic
# preallocation algorithms for, we should do something like:
function _allocates(::typeof(foo), x, y)
    a, bar_allocates = _allocates(bar, x, y)
    b, baz_allocates = _allocates(baz, a, x)
    return b, any((bar_allocates, baz_allocates))
end

# A bit of indirection to make the interface nice.
allocates(::typeof(foo), x, y) = last(_allocates(foo, x, y))
