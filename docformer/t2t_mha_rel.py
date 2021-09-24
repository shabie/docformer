import numpy as np
import tensorflow as tf
import warnings
import torch

warnings.filterwarnings("ignore")


def _generate_relative_positions_matrix_tf(length_q,
                                           length_k,
                                           max_relative_position):
    """Generates matrix of relative positions between inputs."""
    range_vec_q = range_vec_k = tf.range(length_q)
    distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
    distance_mat_clipped = tf.clip_by_value(distance_mat, -max_relative_position, max_relative_position)
    final_mat = distance_mat_clipped + max_relative_position
    return final_mat


def _generate_relative_positions_embeddings_tf(length_q, length_k, depth, max_relative_position):
    relative_positions_matrix = _generate_relative_positions_matrix_tf(length_q, length_k, max_relative_position)
    vocab_size = max_relative_position * 2 + 1
    embeddings_table = tf.compat.v1.get_variable("embeddings", [vocab_size, depth])
    embeddings = tf.gather(embeddings_table, relative_positions_matrix)
    return embeddings


def _generate_relative_positions_matrix_torch(length_q,
                                              length_k,
                                              max_relative_position):
    """Generates matrix of relative positions between inputs."""
    range_vec_q = range_vec_k = torch.arange(length_q)
    distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
    distance_mat_clipped = torch.clamp(distance_mat, -max_relative_position, max_relative_position)
    final_mat = distance_mat_clipped + max_relative_position
    return final_mat


d1 = _generate_relative_positions_matrix_tf(512, 512, 8).numpy()
d2 = _generate_relative_positions_matrix_torch(512, 512, 8).detach().numpy()
np.testing.assert_equal(d1, d2)

d11 = _generate_relative_positions_embeddings_tf(512, 512, 64, 8, "rel_pos").numpy()
print(d11.shape)
