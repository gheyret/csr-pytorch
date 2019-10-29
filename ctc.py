
import torch
from torch import nn
import collections
import numpy
from ctc_decoder import beam_ctc_decode
import timeit





def make_new_beam():
    fn = lambda : (0, 0)
    return collections.defaultdict(fn)


def beam_search(prob_matrix, blank=0, beam_size = 10):
    T, S = prob_matrix.shape
    prob_matrix.detach().numpy()

    beam = [(tuple(), (1.0, 0.0))]

    for t in range(T):

        next_beam = make_new_beam()

        for s in range(S):
            p = prob_matrix[t, s]


            # p_b = P(prefix,t-1) ending in blank
            # p_nb = P(prefix,t-1) not ending in blank
            for prefix, (p_b, p_nb) in beam:


                # new character is blank
                if s == blank:
                    t_p_b, t_p_nb = next_beam[prefix]  # if next_beam doesn't contain prefix a default value is given.
                    t_p_b = t_p_b + (p_b + p_nb)*p  # the first + assures that all paths leading to this are merged.
                    next_beam[prefix] = (t_p_b, t_p_nb)  # COPY previous sequence but change probabilities
                    continue

                # new character isn't blank:
                # new character is equal to the last character:
                last = prefix[-1] if prefix else None
                if s == last:
                    t_p_b, t_p_nb = next_beam[prefix]
                    t_p_nb = t_p_nb + p_nb*p
                    next_beam[prefix] = (t_p_b, t_p_nb)  # COPY previous sequence but change probabilities

                # new character isn't blank:
                # new character isn't equal to the last character:
                prefix_with_new = prefix + (s,)
                t_p_b, t_p_nb = next_beam[prefix_with_new]
                if s != last:  # New character is diff from previous. If prev blank, just add.
                    t_p_nb = t_p_nb + (p_b + p_nb)*p
                else:  # Didn't end in blank, ends in same class as in t-1
                    t_p_nb = t_p_nb + p_b*p

                next_beam[prefix_with_new] = (t_p_b, t_p_nb)  # EXTEND previous sequence with new char

        # Sort and trim the beam before moving on to the
        # next time-step.
        beam = sorted(next_beam.items(),
                      key=lambda x: sum(x[1]),
                      reverse=True)
        beam = beam[:beam_size]
    best = beam[0]
    return best[0], sum(best[1])
T = 25
C = 46
x = torch.randn(T, C)
softmax = nn.Softmax(dim=-1)
logsoftmax = nn.LogSoftmax(dim=-1)
probs = softmax(x)
probsLog = logsoftmax(x)


#labels1, score1 = beam_search(probs)
#labels2, score2 = beam_ctc_decode(probsLog)
#print(labels1)
#print(labels2)
#print(score1)
#print(score2)
#print(-numpy.log(score1))

#print(timeit.timeit("labels1, score1 = beam_search(probs)", number=n, setup="from __main__ import beam_search, probs"))
#print(timeit.timeit("labels2, score2 = beam_ctc_decode(probsLog)", number=n, setup="from __main__ import beam_ctc_decode, probsLog"))
prob_matrix = probs
blank = 0
beam_size = 10

T, S = prob_matrix.shape
#prob_matrix.detach().numpy()

beam = [(tuple(), (1.0, 0.0))]

for t in range(T):

    next_beam = make_new_beam()

    for s in range(S):
        p = prob_matrix[t, s]

        # p_b = P(prefix,t-1) ending in blank
        # p_nb = P(prefix,t-1) not ending in blank
        for prefix, (p_b, p_nb) in beam:

            # new character is blank
            if s == blank:
                t_p_b, t_p_nb = next_beam[prefix]  # if next_beam doesn't contain prefix a default value is given.
                t_p_b = t_p_b + (p_b + p_nb) * p  # the first + assures that all paths leading to this are merged.
                next_beam[prefix] = (t_p_b, t_p_nb)  # COPY previous sequence but change probabilities
                continue

            # new character isn't blank:
            # new character is equal to the last character:
            last = prefix[-1] if prefix else None
            if s == last:
                t_p_b, t_p_nb = next_beam[prefix]
                t_p_nb = t_p_nb + p_nb * p
                next_beam[prefix] = (t_p_b, t_p_nb)  # COPY previous sequence but change probabilities

            # new character isn't blank:
            # new character isn't equal to the last character:
            prefix_with_new = prefix + (s,)
            t_p_b, t_p_nb = next_beam[prefix_with_new]
            if s != last:  # New character is diff from previous. If prev blank, just add.
                t_p_nb = t_p_nb + (p_b + p_nb) * p
            else:  # Didn't end in blank, ends in same class as in t-1
                t_p_nb = t_p_nb + p_b * p

            next_beam[prefix_with_new] = (t_p_b, t_p_nb)  # EXTEND previous sequence with new char

    # Sort and trim the beam before moving on to the
    # next time-step.
    beam = sorted(next_beam.items(),
                  key=lambda x: sum(x[1]),
                  reverse=True)
    beam = beam[:beam_size]
best = beam[0]

beam2 = sorted(next_beam.items(),
              key=lambda x: sum(x[1]),
              reverse=True)



















