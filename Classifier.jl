using Yao, YaoPlots
using StatsBase
using LinearAlgebra
using Base.Threads
using Printf

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
    amplitudes = normalize(amplitudes)
    circuit = prepare(amplitudes, false) 
    push!(circuit, put((m_qubits + 1)=>H))
    push!(circuit, Measure(total_qubits,locs=(m_qubits+1)))  

    # manual setup
    bits = log(2, length(amplitudes))
    new_amplitudes = copy(amplitudes)
    for i in 0:(length(amplitudes) - 1)
        i_r = binary_reverse(bits, i)
        new_amplitudes[i + 1] = amplitudes[i_r + 1] # the indexes are interchangable
    end
    amplitudes = new_amplitudes
    cheese = ArrayReg(complex(amplitudes))
    test = zero_state(total_qubits)
    apply!(test, circuit)
    aux_circ = chain(total_qubits, put((m_qubits + 1)=>H), Measure(total_qubits,locs=(m_qubits+1)))
    #apply!(cheese, aux_circ)
    #print(statevec(cheese), "\n\n", statevec(test), "\n\n", test == cheese, "\n\n")

    # simulation
    m_circ = chain(total_qubits, Measure(total_qubits, locs=total_qubits))
    prediction = 0.0
    fails = 0
    for epoch in 1:epochs
        # post selection
        #state_ = zero_state(total_qubits)
        #apply!(state_, circuit)
        state_ = copy(cheese)
        apply!(state_, aux_circ)
        while measure(state_, m_qubits+1)[1][1] != 1
            fails += 1
            #state_ = zero_state(total_qubits)
            #apply!(state_, circuit)
            state_=copy(cheese)
            apply!(state_, aux_circ)
        end

        # measurement
        prediction += measure(apply(state_, m_circ),total_qubits)[1][1]
    end

    print(@sprintf("Post selection failures: %d of %d epochs| ratio: %.3f\n", fails, epochs, fails / (epochs + fails)))

    return 1 - prediction / epochs
end
