import subprocess

# Step 1
print("Running generate_for_eval.py")
subprocess.run(["python", "generate_for_eval.py", "--model", "zjhhhh/ultrafeedback_bon_rebel_555134_1750278713", 
                "--output_repo", "MisDrifter/eval_bon_rebel", "--start_idx", "3000", 
                "--end_idx", "4000", "--world_size", "1"], check=True)

# Step 2
print("Running rank_for_eval.py")
subprocess.run(["python", "rank_for_eval.py", "--input_repo", "MisDrifter/eval_bon_rebel", "--selection_pairs", "2"], check=True)

# Step 3
print("Running eval_stats.py")
subprocess.run(["python", "eval_stats.py", "--input_repo", "MisDrifter/eval_bon_rebel_armo"], check=True)
