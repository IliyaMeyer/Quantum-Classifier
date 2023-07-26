using Yao, YaoPlots
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
    circuit = prepare(amplitudes, false) #TODO I don't even know
    push!(circuit, put((m_qubits + 1)=>H))
    push!(circuit, Measure(total_qubits,locs=(m_qubits+1)))  

    # simulation
    m_circ = chain(total_qubits, Measure(total_qubits, locs=total_qubits))
    prediction = 0.0
    for epoch in 1:epochs
        # post selection
        state_ = zero_state(total_qubits)
        r = apply!(state_, circuit)
        while measure(r, m_qubits+1)[1][1] != 1
            state_ = zero_state(total_qubits)
            r = apply!(state_, circuit)
        end

        # measurement
        r = apply!(state_, m_circ)
        measurement = measure(r,total_qubits)
        prediction += measurement[1][1]
    end

    return 1 - prediction / epochs
end
