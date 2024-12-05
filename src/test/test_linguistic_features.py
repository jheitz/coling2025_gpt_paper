from preprocessing.linguistic_features import LinguisticFeatures

class TestLinguisticFeatures:
    linguistic_features = LinguisticFeatures(config={}, constants=None)
    def test_coordinate_subordinate_clauses(self):
        # Make sure you use stanza 1.5 for this to work (no longer works in stanza 1.7)
        def do_test(text, expected_result):
            doc = self.linguistic_features.nlp_pipeline(text)
            n_coordinate, n_subordinate, fraction = self.linguistic_features._count_coordinate_subordinate_clauses(doc)
            result = (n_coordinate, n_subordinate)
            assert result == expected_result, \
                f"Wrong result for '{text}': {result}, but expecting {expected_result}"

        ## these work fine
        do_test('The boy is standing on the stool while the mother is washing dishes.', (0, 1))
        do_test('The boy is standing on the stool and the mother is washing dishes.', (1, 0))
        do_test('The boy is standing on the stool but the mother is washing dishes and the water is overflowing.',
                (2, 0))
        do_test("I don't think he's stealing", (0, 1))
        do_test("I see two kids up at the cookie jar, one on a stool the other standing on the floor", (0, 0))
        do_test("the kitchen sink is running over, the water running out on the floor", (0, 0))
        do_test("the kitchen sink is running over, and the water running out on the floor", (1, 0))

        ## These cause some problems
        # the next sentence is parsed to have 5 subordinate clauses, check out the graph:
        # check it out on http://brenocon.com/parseviz/ using the following parse tree:
        # (ROOT (S (CC and) (UH uh) (NP (DT that)) (VP (VBZ 's) (NP (DT the) (UH uh)) (NP (DT the) (UH uh)) (PP (UH oh) (NP (DT the) (NN bench))) (CC or) (SBAR (SBAR (WHNP (WP what)) (S (SQ (VBP do) (S (NP (PRP you)) (VP (VB call) (NP (PRP it)) (ADVP (UH uh))))))) (VP (VBZ is) (SBAR (S (NP (PRP it)) (VP (VBZ looks) (SBAR (IN like) (S (NP (PRP it)) (VP (VBZ 's) (ADVP (RB gonna)) (VP (VB fall))))))))))) (. .)))
        do_test("and uh that's the uh the uh oh the bench or what do you call it uh is it looks like it's gonna fall",
                (0, 3))

        # the next are two sentences, but should be one (?) -> result of the utterance segmentation in CHAT transcriptions
        # as a result, it's not counted as a coordinate clause, should it?
        do_test("he's gonna fall down off the ladder . and the mother's washing the dishes", (0, 0))

        # "looks like" is parsed as a subordinate clause, should it really?
        do_test("looks like the mother is doing washing the dishes", (0, 1))