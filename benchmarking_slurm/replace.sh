#!/bin/bash

# Directory containing your SLURM job scripts
SCRIPT_DIR="."  # Change this if your scripts are in a different directory

# Loop over all SLURM scripts in the directory (adjust the pattern as needed)
for script in "$SCRIPT_DIR"/*.slurm; do
    # Create a temporary file to store the modified script
    tmp_file=$(mktemp)

    while IFS= read -r line; do
        if [[ $line == srun* ]]; then
            echo "cd .." >> "$tmp_file"
            echo "${line/..\/run_gen_benchmark_westpa.sh/.\/run_gen_benchmark_westpa.sh}" >> "$tmp_file"
        else
            echo "$line" >> "$tmp_file"
        fi
    done < "$script"

    # Overwrite the original script with the modified content
    mv "$tmp_file" "$script"
    echo "Updated $script"
done

