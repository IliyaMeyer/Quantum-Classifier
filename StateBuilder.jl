using Yao
using StatsBase
using LinearAlgebra
using Base.Threads

include("Utils.jl")

function prepare(amplitudes, computational_basis)

    # prepare amplitudes vector
    amplitudes = copy(amplitudes)
    while log(2, length(amplitudes)) != ceil(log(2, length(amplitudes)))
        push!(amplitudes, 0)
    end
    amplitudes = normalize(amplitudes)
    
    # reorder states vector for endian purposes (makes bitwise operations convienient)
    bits = log(2, length(amplitudes))
    if computational_basis
        new_amplitudes = copy(amplitudes)
        for i in 0:(length(amplitudes) - 1)
            i_r = binary_reverse(bits, i)
            new_amplitudes[i + 1] = amplitudes[i_r + 1] # the indexes are interchangable
        end
        amplitudes = new_amplitudes
    end
    
    # U-gate from Long-Sun paper
    function U(a)
        if isnan(a)
            a = 0
        end
        return chain(1, put(1=>Z), put(1=>Ry(2*a)))
    end
    
    # initialize  circuit
    num_qubits = round(Int, log(2, length(amplitudes)))
    circuit = chain(num_qubits)
    circuit_state = []
    
    # X gates
    function update_state()
        state_index = length(circuit_state)
        while state_index > 0
            if circuit_state[state_index] == 0
                circuit_state[state_index] = 1
                push!(circuit, put(state_index=>X))
                break
            else
                circuit_state[state_index] = 0
                push!(circuit, put(state_index=>X))
                if state_index == 1
                    break
                else
                    state_index -= 1
                end
            end 
        end
    end
    
    # return the value in the states array
    function a(index)
        return amplitudes[index + 1]
    end
    
    # return the angle α_j,i_1 i_2 ...
    function alpha(qubit, index)
        numerator_index = ((index << 1) + 1) << (num_qubits - qubit)
        denominator_index = (index << 1) << (num_qubits - qubit)
        numerator = 0
        denominator = 0
        for i in 0:(2^(num_qubits - qubit) - 1)
            numerator += abs(a(numerator_index))^2
            denominator += abs(a(denominator_index))^2
            numerator_index += 1
            denominator_index += 1
        end
        return atan(sqrt(numerator/denominator))
    end
    
    # work out alphas
    alphas = zeros(2^num_qubits - 1)
    for qubit in 1:num_qubits
        @threads for i in (2^(qubit-1)):(2^qubit - 1)
            alphas[i] = alpha(qubit, i - 2^(qubit-1))
        end
    end

    # setup states for each qubit
    processed_qubits = []
    for qubit in 1:num_qubits
        # build and add each of the rotation gates
        for i in (2^(qubit-1)):(2^qubit - 1)
            update_state()
            push!(circuit, control(processed_qubits,qubit=>U(alphas[i])))
        end
        push!(processed_qubits, qubit)
        push!(circuit_state, 1)
    end
    
    return circuit
end
