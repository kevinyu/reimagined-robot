import theano
import theano.tensor as T

import config

from glimpse import accumulate_glimpses
from utils.complex import complex_multiply, complex_conj, complex_map, complex_dot


def unbind_sample_locations(S, sample_positions):
    """From a several scene representations, unbind positions at sample locations

    Params:
    S (complex tuple)
        0, 1: (T.fmatrix, N_glimpses x N_hyper)
            real and imaginary parts of scene representations over glimpses
            each column of matrix is state of scene after n glimpses, S[n]
    sample_positions (complex tuple)
        0, 1: (T.fmatrix, N_samples x N_hyper)
            real and imaginary part of locations where scene has been sampled

    Returns:
    S_unbound (complex tuple)
        0, 1: (T.ftensor3, N_glimpses x N_samples x N_hyper)
            the real and imaginary parts of hypervectors representing the scene
            with each sample location unbound;
            S_unbound[i, j, :] is the scene hypervector after i glimpses,
            multiplied by the complex conjugate (inverse) of sample j's position vector
    """
    S = complex_map(S, lambda s: s.dimshuffle(0, "x", 1))
    sample_positions = complex_map(sample_positions, lambda s: s.dimshuffle("x", 0, 1))

    return complex_multiply(S, complex_conj(sample_positions))


def evaluate_digit_similarities(S, digits):
    """Evaluate simlarity of tensor of scene representations to a set of digits

    S (complex tuple)
        0, 1: (T.ftensor3, N_glimpses x N_samples x N_hyper)
            S[i, j, :] is a hypervector to evaluate the presence of a digit
    digits (complex tuple):
        0, 1: (T.fmatrix, N_hyper x N_digits)
            Matrix whose columns are hypervectors representing digits themselves

    Returns:
    similarities (complex tuple)
        0, 1: (T.fmatrix, N_glimpses x N_samples x N_digits)
            Matrix whose elements [i, j, k] represent the degree of similarity (using the dot product)
            between the scene representation after glimpse i at location j, to digit k
            Only the real part of the dot product is taken
    """
    return complex_dot(S, digits)[0]  # real part!


def apply_softmax_to_digit_similarities(similarities):
    """Similarities
  
    Params:
    similarities (T.fmatrix, N_glimpses x N_samples x N_digits)
    
    Returns:
    softmax_output (T.fmatrix, N_glimpses x N_samples x N_digits)
    """
    a, b, c = similarities.shape
    return T.nnet.softmax(similarities.reshape((a * b, c))).reshape(similarities.shape)


def apply_categorical_crossentropy(softmax_output, sample_labels):
    """
    """
    a, b, c = softmax_output.shape
    return T.nnet.categorical_crossentropy(
            softmax_output,
            sample_labels
    )


def compute_cost_of_scene_over_glimpses(
        glimpse_vectors_real,
        glimpse_vectors_imag,
        glimpse_positions_real,
        glimpse_positions_imag,
        sample_labels,
        sample_positions_real,
        sample_positions_imag,
        cost,
        S0_real,
        S0_imag,
        digits_real,
        digits_imag):

    S0 = (S0_real, S0_imag)
    digits = (digits_real, digits_imag)
    glimpse_vectors = (glimpse_vectors_real, glimpse_vectors_imag)
    glimpse_positions = (glimpse_positions_real, glimpse_positions_imag)
    sample_positions = (sample_positions_real, sample_positions_imag)

    S = accumulate_glimpses(S0, glimpse_vectors, glimpse_positions)
    S = unbind_sample_locations(S, sample_positions)
    sim = evaluate_digit_similarities(S, digits)
    softmax_output = apply_softmax_to_digit_similarities(sim)

    return cost + apply_categorical_crossentropy(
            softmax_output,
            sample_labels.dimshuffle("x", 0, 1)).mean()


def compute_cost_of_batch(
        S0,
        batch_glimpse_vectors,  # N_batches x 
        batch_glimpse_positions,  # N_batches x N_hyper x N_glimpses
        batch_sample_labels,  # N_batches x N_samples x N_digits
        batch_sample_positions,  # N_batches x N_hyper x N_samples
        digits):
   
    cost = T.constant(0.0, dtype=theano.config.floatX)
    result, updates = theano.scan(
            fn=compute_cost_of_scene_over_glimpses,
            outputs_info=cost,
            sequences=[
                batch_glimpse_vectors[0],
                batch_glimpse_vectors[1],
                batch_glimpse_positions[0],
                batch_glimpse_positions[1],
                batch_sample_labels,
                batch_sample_positions[0],
                batch_sample_positions[1]
            ],
            non_sequences=[
                S0[0],
                S0[1],
                digits[0],
                digits[1],
            ]
    )
    return result[-1]

