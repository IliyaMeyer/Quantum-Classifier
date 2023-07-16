using Yao
using StatsBase
using LinearAlgebra
using Base.Threads

include("Utils.jl")
include("StateBuilder.jl")

function classify(x, y, x_unk, epochs)
    num_records = size(x, 1)
    num_features = size(x, 2)
    feature_qubits = round(Int,ceil(log(2, num_features)))
    m_qubits = round(Int,ceil(log(2, num_records))) 
    y_qubits = 1
    total_qubits = m_qubits + 1 + feature_qubits + y_qubits

    # build the amplitudes
    amplitudes = zeros(2^total_qubits)
    for m in 0:(2^m_qubits - 1) # for each data point
        for i in 0:(2^feature_qubits - 1) # for each feature
            classification_bit = y[m+1]
            # unknown point
            amplitudes[((((m << 1) << feature_qubits) + i) << y_qubits) + classification_bit + 1] = x_unk[i+1] / sqrt(2 * num_records)
            # data point
            amplitudes[(((((m<<1)+1)<<feature_qubits)+i)<<y_qubits)+classification_bit + 1] = x[m+1,i+1] / sqrt(2 * num_records)
        end
    end

    # build circuit
    circuit = prepare(amplitudes, false)
    push!(circuit, put((m_qubits + 1)=>H))
    push!(circuit, Measure(total_qubits,locs=(m_qubits+1)))
    push!(circuit, Measure(total_qubits,locs=(total_qubits)))

    probs = [0.0, 0.0]
    @threads for i in 1:epochs
        # simulate
        state = zero_state(total_qubits)
        apply!(state, circuit)

        # check for which state has the greatest probability
        for i in 1:2^total_qubits
            probs[(binary_reverse(total_qubits,i)&1) + 1] += abs(statevec(state)[i])^2
        end
    end
    probs ./= epochs

    return argmax(probs) - 1
end
