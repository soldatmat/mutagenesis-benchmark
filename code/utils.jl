using DataFrames

recombine_symbols(sequence_length::Int, alphabet::Set{Char}) = map(sequence -> collect(sequence), collect(Iterators.product(ntuple(_ -> alphabet, sequence_length)...))[:])

function _get_variants(data_path::String, csv_file::String)
    variants = CSV.read(joinpath(data_path, csv_file), DataFrame)
    [collect(values(row)[1]) for row in eachrow(variants)]
end
_construct_sequence(variant::Vector{Char}, wt_string::String, mutation_positions::Vector{Int}) = collect(wt_string[1:mutation_positions[1]-1] * variant[1] * wt_string[mutation_positions[1]+1:mutation_positions[2]-1] * variant[2] * wt_string[mutation_positions[2]+1:mutation_positions[3]-1] * variant[3] * wt_string[mutation_positions[3]+1:mutation_positions[4]-1] * variant[4] * wt_string[mutation_positions[4]+1:end])
_construct_sequences(variants::AbstractVector{Vector{Char}}, wt_string::String, mutation_positions::Vector{Int}) = map(v -> _construct_sequence(v, wt_string, mutation_positions), variants)
_get_sequences(data_path::String, csv_file::String, wt_string::String, mutation_positions::Vector{Int}) = _construct_sequences(_get_variants(data_path, csv_file), wt_string, mutation_positions)

function _get_fitness(data_path::String, csv_file::String)
    fitness = CSV.read(joinpath(data_path, csv_file), DataFrame)
    fitness = [values(row)[1] for row in eachrow(fitness)]
end

extract_residues(sequence::AbstractVector{Char}, mutation_posisitions::AbstractVector{Int}) = map(pos -> sequence[pos], mutation_posisitions)
