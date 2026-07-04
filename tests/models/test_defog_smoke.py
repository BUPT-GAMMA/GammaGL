import os
import subprocess
import sys

def test_defog_smoke():
    """
    Smoke test to verify that the DeFoG training loop runs without errors
    on a tiny synthetic dataset. This ensures that the core components
    (model, loss, flow matching, dataloader) work properly on CPU and do not
    crash due to missing heavy dependencies (like RDKit or Graph-Tool).
    """
    # Assuming the test is run from the repository root
    script_path = os.path.join("examples", "defog", "defog_trainer.py")
    if not os.path.exists(script_path):
        # Fallback if tests are run from a different working directory
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        script_path = os.path.join(repo_root, "examples", "defog", "defog_trainer.py")

    assert os.path.exists(script_path), f"Cannot find trainer script at {script_path}"

    cmd = [
        sys.executable,
        script_path,
        "--dataset", "synthetic",
        "--num_graphs", "4",
        "--min_nodes", "4",
        "--max_nodes", "8",
        "--batch_size", "2",
        "--n_epochs", "1",
        "--n_layers", "2",
        "--hidden_mlp_X", "16",
        "--hidden_mlp_E", "8",
        "--hidden_mlp_y", "16",
        "--dx", "16",
        "--de", "8",
        "--dy", "16",
        "--dim_ffX", "32",
        "--dim_ffy", "32",
        "--check_val_every_n_epochs", "1",
        "--gpu", "-1",
        "--data_root", "./_review_data",
        "--save_dir", "./_review_outputs"
    ]

    env = os.environ.copy()
    env['TL_BACKEND'] = 'torch'

    print("Running DeFoG smoke test...")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Stdout:\n", result.stdout)
        print("Stderr:\n", result.stderr)
        
    assert result.returncode == 0, f"DeFoG smoke test failed with return code {result.returncode}"
