# tests/test_selection_weights.py
# from guca.ga.toolbox import _rank_weights, _roulette_weights

# def test_rank_weights_strictly_descend():
#     w = _rank_weights(5)  # [5,4,3,2,1] / sum => probs
#     assert len(w) == 5
#     # monotone strictly decreasing
#     for i in range(4):
#         assert w[i] > w[i+1]
#     # normalized
#     assert abs(sum(w) - 1.0) < 1e-12

# def test_roulette_weights_shifts_nonpositive():
#     # includes zero/negative; should shift and normalize
#     fits = [0.0, -2.0, 3.0, 9.0]
#     w = _roulette_weights(fits)
#     assert len(w) == 4
#     assert all(p >= 0 for p in w)
#     assert abs(sum(w) - 1.0) < 1e-12
#     # largest fitness gets largest probability
#     assert w[-1] == max(w)
