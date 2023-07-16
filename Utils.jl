function binary_reverse(num_bits, i)
    i_r = 0
    for j in 1:num_bits
        i_r = i_r << 1
        i_r += i & 1
        i = i >> 1
    end
    return i_r
end
