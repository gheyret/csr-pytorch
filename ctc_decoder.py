import torch
import numpy
import math
import collections
import Levenshtein


def collapse_sequence(input_sequence, blank_code):
    collapsed_sequence = [input_sequence[i] for i in range(len(input_sequence)) if (i==0) or input_sequence[i] != input_sequence[i-1]]
    filtered_sequence = [value for value in collapsed_sequence if value != blank_code]
    return filtered_sequence


def greedy_decode_ctc(input_sample, blank_code):
    top_indices = input_sample.topk(k=1,dim=2)[1]
    decoded_sequence = numpy.asarray(top_indices.view(-1))
    output_sequence = collapse_sequence(decoded_sequence, blank_code)

    return output_sequence


NEG_INF = -float("inf")


def make_new_beam():
    fn = lambda : (NEG_INF, NEG_INF)
    return collections.defaultdict(fn)


def logsumexp(*args):
    """
    Stable log sum exp.
    """
    if all(a == NEG_INF for a in args):
        return NEG_INF
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max)
                       for a in args))
    return a_max + lsp


def decode_sample(input_batch, sample_id):
    input_batch = input_batch.cpu()
    probability_matrix = input_batch[:, sample_id, :].detach().numpy()
    labels, score = beam_ctc_decode(probability_matrix)
    return labels


def compute_edit_distance(output_batch, targets, target_lengths, num_samples=0):
    if num_samples > 0:
        total_edit_distance = 0
        for i in range(0, num_samples):
            label = decode_sample(output_batch, i)
            s1 = [chr(x) for x in label]
            s2 = [chr(x) for x in targets[i, :target_lengths[i]]]
            distance = Levenshtein.distance(''.join(s1), ''.join(s2))
            edit_distance = distance/len(s2)
            total_edit_distance += edit_distance
        avg_edit_distance = total_edit_distance/num_samples
        return avg_edit_distance
    else:
        return 0.0


def beam_ctc_decode(probs, beam_size=10, blank=0):
    """
    Performs inference for the given output probabilities.
    Arguments:
      probs: The output probabilities (e.g. post-softmax) for each
        time step. Should be an array of shape (time x output dim).
      beam_size (int): Size of the beam to use during inference.
      blank (int): Index of the CTC blank label.
    Returns the output label sequence and the corresponding negative
    log-likelihood estimated by the decoder.
    """
    T, S = probs.shape
    #probs = numpy.log(probs)

    # Elements in the beam are (prefix, (p_blank, p_no_blank))
    # Initialize the beam with the empty sequence, a probability of
    # 1 for ending in blank and zero for ending in non-blank
    # (in log space).
    beam = [(tuple(), (0.0, NEG_INF))]

    for t in range(T): # Loop over time

        # A default dictionary to store the next step candidates.
        next_beam = make_new_beam()

        for s in range(S): # Loop over vocab
            p = probs[t, s]

            # The variables p_b and p_nb are respectively the
            # probabilities for the prefix given that it ends in a
            # blank and does not end in a blank at this time step.
            for prefix, (p_b, p_nb) in beam: # Loop over beam

                # If we propose a blank the prefix doesn't change.
                # Only the probability of ending in blank gets updated.
                if s == blank:
                    n_p_b, n_p_nb = next_beam[prefix]
                    n_p_b = logsumexp(n_p_b, p_b + p, p_nb + p)
                    next_beam[prefix] = (n_p_b, n_p_nb)
                    continue

                # Extend the prefix by the new character s and add it to
                # the beam. Only the probability of not ending in blank
                # gets updated.
                end_t = prefix[-1] if prefix else None
                n_prefix = prefix + (s,)
                n_p_b, n_p_nb = next_beam[n_prefix]
                if s != end_t:
                    n_p_nb = logsumexp(n_p_nb, p_b + p, p_nb + p)
                else:
                    # We don't include the previous probability of not ending
                    # in blank (p_nb) if s is repeated at the end. The CTC
                    # algorithm merges characters not separated by a blank.
                    n_p_nb = logsumexp(n_p_nb, p_b + p)

                # *NB* this would be a good place to include an LM score.
                next_beam[n_prefix] = (n_p_b, n_p_nb)

                # If s is repeated at the end we also update the unchanged
                # prefix. This is the merging case.
                if s == end_t:
                    n_p_b, n_p_nb = next_beam[prefix]
                    n_p_nb = logsumexp(n_p_nb, p_nb + p)
                    next_beam[prefix] = (n_p_b, n_p_nb)

        # Sort and trim the beam before moving on to the
        # next time-step.
        beam = sorted(next_beam.items(),
                key=lambda x : logsumexp(*x[1]),
                reverse=True)
        beam = beam[:beam_size]

    best = beam[0]
    return best[0], -logsumexp(*best[1])