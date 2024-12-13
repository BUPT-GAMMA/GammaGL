import numpy as np
import tempfile
from gammagl.utils.read_embeddings import read_embeddings


def test_read_embeddings():
    # Create a temporary file with example embeddings
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmpfile:
        tmpfile.write("0 0.1 0.2 0.3\n")
        tmpfile.write("1 0.4 0.5 0.6\n")
        tmpfile.write("2 0.7 0.8 0.9\n")
        tmpfile_name = tmpfile.name

    # Expected embedding matrix
    expected_embedding_matrix = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ])

    # Read embeddings using the function
    embedding_matrix = read_embeddings(tmpfile_name, 3, 3)

    # Verify the embedding matrix using assert
    assert np.allclose(embedding_matrix, expected_embedding_matrix), f"Expected: {expected_embedding_matrix}, but got: {embedding_matrix}"

    # Clean up the temporary file
    import os
    os.remove(tmpfile_name)
    print("Test passed!")
