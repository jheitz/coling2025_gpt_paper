import os
import json
import stanza
from collections import defaultdict
import pandas as pd
import numpy as np
from functools import reduce
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI
import time
from datetime import datetime

from preprocessing.preprocessor import Preprocessor
from dataloader.dataset import TextDataset, TabularDataset
from util.helpers import safe_divide, dataset_name_to_url_part, create_directory
from util.decorators import cache_to_file_decorator



class LinguisticFeatureLiteratureCalculator:

    def __init__(self, doc, constants=None):
        self.CONSTANTS = constants
        self.doc = doc
        self.words_enriched = [word for sent in doc.sentences for word in sent.words]
        self.words_raw = [word.text for sent in doc.sentences for word in sent.words]
        self.sentences_enriched = doc.sentences
        self.sentences_raw = [[w.text for w in sentence.words] for sentence in doc.sentences]


    def n_words(self):
        """ Number of words """
        return len(self.words_raw)

    def n_unique(self):
        """ Number of unique words / types """
        return len(set(self.words_raw))

    def word_length(self):
        """ Average length of words in letters """
        return safe_divide(sum([len(word) for word in self.words_raw]), self.n_words())

    def sentence_length(self):
        """ Average length of sentences in words """
        return safe_divide(sum([len(sentence) for sentence in self.sentences_raw]), len(self.sentences_raw))

    def pos_counts(self):
        """
        Count number of POS tags in doc
        """
        # universal POS (UPOS) tags (https://universaldependencies.org/u/pos/)
        count_pos = defaultdict(lambda: 0)
        # treebank-specific POS (XPOS) tags (https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)
        count_xpos = defaultdict(lambda: 0)
        for word in self.words_enriched:
            count_pos[word.pos] += 1
            count_xpos[word.xpos] += 1

        return count_pos, count_xpos

    def constituency_rules_count(self):
        """
        Returns counts of all CFG production rules from the constituency parsing of the doc
        Check http://surdeanu.cs.arizona.edu/mihai/teaching/ista555-fall13/readings/PennTreebankConstituents.html
        for available tags
        """
        def constituency_rules(doc):
            def get_constituency_rule_recursive(tree):
                label = tree.label
                children = tree.children
                if len(children) == 0:  # this is a leaf
                    return []

                # find children label, if child is not a leaf. also ignore full stops
                children_label = "_".join([c.label for c in children if len(c.children) > 0 and c.label != '.'])
                if len(children_label) > 0:
                    rule = f"{label} -> {children_label}"
                    # print(f"{rule}     ({tree})")
                else:
                    return []

                children_rules = [get_constituency_rule_recursive(c) for c in children]
                children_rules_flat = [ll for l in children_rules for ll in l]

                return children_rules_flat + [rule]

            all_rules = []
            for sentence in doc.sentences:
                tree = sentence.constituency
                rules = get_constituency_rule_recursive(tree)
                all_rules += rules

            return all_rules

        cfg_rules = constituency_rules(self.doc)
        count_rules = defaultdict(lambda: 0)
        for rule in cfg_rules:
            count_rules[rule] += 1

        return count_rules

    def get_constituents(self):
        """
        Get constituents of the document, with corresponding text (concatenation of the leaf words)
        Check http://surdeanu.cs.arizona.edu/mihai/teaching/ista555-fall13/readings/PennTreebankConstituents.html
        for available tags
        """
        def get_leafs_text(tree):
            if len(tree.children) == 0:
                return tree.label
            return " ".join([get_leafs_text(c) for c in tree.children])

        def get_constituents_recursive(tree):
            label = tree.label
            if len(tree.children) == 0:
                return []

            children_constituents = [get_constituents_recursive(c) for c in tree.children]
            children_constituents_flat = [ll for l in children_constituents for ll in l]

            text = get_leafs_text(tree)
            return children_constituents_flat + [{'label': label, 'text': text}]

        all_constituents = []
        for sentence in self.sentences_enriched:
            tree = sentence.constituency
            rules = get_constituents_recursive(tree)
            all_constituents += rules
        return all_constituents

    def mattr(self, window_length=20):
        """
        Moving-Average Type-Token Ratio (MATTR)
        Adapted from https://github.com/kristopherkyle/lexical_diversity/blob/master/lexical_diversity/lex_div.py
        """
        if len(self.words_raw) == 0:
            return 0
        elif len(self.words_raw) < (window_length + 1):
            ma_ttr = len(set(self.words_raw)) / len(self.words_raw)
        else:
            sum_ttr = 0
            for x in range(len(self.words_raw) - window_length + 1):
                window = self.words_raw[x:(x + window_length)]
                sum_ttr += len(set(window)) / float(window_length)
            ma_ttr = sum_ttr / len(self.words_raw)
        return ma_ttr

    def ttr(self):
        """
        Type-token ratio
        Adapted from https://github.com/kristopherkyle/lexical_diversity/blob/master/lexical_diversity/lex_div.py
        """
        ntokens = len(self.words_raw)
        ntypes = len(set(self.words_raw))
        return safe_divide(ntypes, ntokens)

    def brunets_indes(self):
        """
        Brunét's index, based on definition in https://aclanthology.org/2020.lrec-1.176.pdf
        """
        n_tokens = len(self.words_raw)
        n_types = len(set(self.words_raw))
        if n_tokens > 0:
            return n_tokens ** n_types ** (-0.165)
        else:
            return 0

    def honores_statistic(self):
        """
        Honoré's statistic, based on definition in https://aclanthology.org/2020.lrec-1.176.pdf
        """
        n_words_with_one_occurence = len(list(filter(lambda w: self.words_raw.count(w) == 1, self.words_raw)))
        n_words = len(self.words_raw)
        n_types = len(set(self.words_raw))
        if n_words == 0:
            return 0
        elif (1 - n_words_with_one_occurence / n_types) == 0:
            return 0
        else:
            return (100 * np.log(n_words)) / (1 - n_words_with_one_occurence / n_types)


    def _count_syllables(self, word):
        """
        Simple syllable counting, from on  https://github.com/cdimascio/py-readability-metrics/blob/master/readability/text/syllables.py
        """
        word = word if type(word) is str else str(word)
        word = word.lower()
        if len(word) <= 3:
            return 1
        word = re.sub('(?:[^laeiouy]es|[^laeiouy]e)$', '', word)  # removed ed|
        word = re.sub('^y', '', word)
        matches = re.findall('[aeiouy]{1,2}', word)
        return len(matches)

    def flesch_kincaid(self):
        """
        Flesch–Kincaid grade level
        Based on https://github.com/cdimascio/py-readability-metrics/blob/master/readability/scorers/flesch_kincaid.py
        Formula adapted according to https://en.wikipedia.org/wiki/Flesch–Kincaid_readability_tests
        """
        words = [word for sent in self.sentences_raw for word in sent]

        syllables_per_word = [self._count_syllables(w) for w in words]
        avg_syllables_per_word = safe_divide(sum(syllables_per_word), len(words))

        avg_words_per_sentence = safe_divide(len(words), len(self.sentences_raw))

        return (0.39 * avg_words_per_sentence + 11.8 * avg_syllables_per_word) - 15.59

    @cache_to_file_decorator()
    def _cos_dist_between_sentences(self, sentences_raw):
        """
        Average cosine distance between utterances, and proportion of sentence pairs whose cosine distance is less than or equal to 0.5
        Based on and adapted from https://github.com/vmasrani/dementia_classifier/blob/1f48dc89da968a6c9a4545e27b162c603eb9a310/dementia_classifier/feature_extraction/feature_sets/psycholinguistic.py#L686
        """
        stop = nltk.corpus.stopwords.words('english')
        stemmer = nltk.PorterStemmer()
        def not_only_stopwords(text):
            unstopped = [w for w in text.lower() if w not in stop]
            return len(unstopped) != 0
        def stem_tokens(tokens):
            return [stemmer.stem(item) for item in tokens]
        def normalize(text):
            text = str(text).lower()
            return stem_tokens(nltk.word_tokenize(text))

        def cosine_sim(text1, text2):
            # input: list of raw utterances
            # returns: list of cosine similarity between all pairs
            if not_only_stopwords(text1) and not_only_stopwords(text2):
                # Tfid raises error if text contain only stopwords. Their stopword set is different
                # than ours so add try/catch block for strange cases
                try:
                    vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')  # Punctuation remover
                    tfidf = vectorizer.fit_transform([text1, text2])
                    return ((tfidf * tfidf.T).A)[0, 1]
                except ValueError as e:
                    print("Error:", e)
                    print('Returning 0 for cos_sim between: "', text1, '" and: "', text2, '"')
                    return 0
            else:
                return 0

        def compare_all_utterances(uttrs):
            # input: list of raw utterances
            # returns: (float)average similarity over all similarities
            similarities = []
            for i in range(len(uttrs)):
                for j in range(i + 1, len(uttrs)):
                    similarities.append(cosine_sim(uttrs[i], uttrs[j]))
            return similarities

        def avg_cos_dist(similarities):
            # returns:(float) Minimum similarity over all similarities
            return safe_divide(reduce(lambda x, y: x + y, similarities), len(similarities))

        def proportion_below_threshold(similarities, thresh):
            # returns: proportion of sentence pairs whose cosine distance is less than or equal to a threshold
            valid = [s for s in similarities if s <= thresh]
            return safe_divide(len(valid), float(len(similarities)))

        if len(sentences_raw) < 2:
            # only one sentence -> return 0
            return 0, 0
        sentences_as_text = [" ".join(sentence) for sentence in sentences_raw]
        similarities = compare_all_utterances(sentences_as_text)
        return avg_cos_dist(similarities), proportion_below_threshold(similarities, 0.5)

    def cos_dist_between_sentences(self):
        return self._cos_dist_between_sentences(self.sentences_raw)


    @cache_to_file_decorator(verbose=False)
    def get_english_dictionary(self):
        # word list from spell checker library
        # https://github.com/barrust/pyspellchecker/blob/master/spellchecker/resources/en.json.gz
        with open(os.path.join(self.CONSTANTS.RESOURCES_DIR, 'english_dictionary.csv')) as f:
            words = f.readlines()
            words = [w.strip() for w in words]
            # add some words not in dictionary but okay, such as n't token from "don't"
            words += ["n't", "'re", "'ll"]
            return words

    def not_in_dictionary(self):
        words_greater_than_two = [w for w in self.words_raw if len(w) > 2]  # only consider words greater than length 2
        #not_found = [w for w in words_greater_than_two if w.lower() not in nltk.corpus.words.words()]
        not_found = [w for w in words_greater_than_two if w.lower() not in self.get_english_dictionary()]
        print("Words not in dictionary: ", not_found)
        return safe_divide(len(not_found), len(words_greater_than_two))



class GPTFeatureExtractor:

    def __init__(self, version, gpt_model=None, run_parameters=None):
        self.version = version
        self.client = OpenAI(api_key=open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../keys/openai-key.txt"), "r").read())
        self.temperature = 0
        self.run_parameters = run_parameters
        self.seed = 1234
        self.gpt_model = gpt_model if gpt_model is not None else 'gpt-4-1106-preview'

    def _get_prompt_templates(self, prompt_postfix=""):
        if self.version == '5features':
            with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "gpt_feature_prompts/5features/system.txt"), "r") as f:
                system_template = f.read()
            with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), f"gpt_feature_prompts/5features/prompt{prompt_postfix}.txt"), "r") as f:
                prompt_template = f.read()
        elif self.version == '10features':
            with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "gpt_feature_prompts/10features/system.txt"), "r") as f:
                system_template = f.read()
            with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), f"gpt_feature_prompts/10features/prompt{prompt_postfix}.txt"), "r") as f:
                prompt_template = f.read()
        else:
            raise ValueError("Invalid GPT prompt version", self.version)
        return system_template, prompt_template

    def _log_api_call(self, data):
        with open(os.path.join(self.run_parameters.results_dir, "openai_api_log.txt"), "a") as f:
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{time}: {json.dumps(data)}\n\n")

    @cache_to_file_decorator(n_days=365)  # only recalculate after 1 year
    def _call_openai_api(self, identifier, system, prompt, temperature=0, seed=123, model='gpt-4-1106-preview'):
        print(f"Calling OpenAI API for GPT features version {self.version} of transcript {identifier} ({model}, temp={temperature}, seed={seed})...")
        completions_parameters = dict(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            top_p=1,
            seed=seed,
        )
        start_time = time.time()
        response = self.client.chat.completions.create(**completions_parameters)
        log_data = {
            'identifier': identifier,
            **completions_parameters,
            'response_content': response.choices[0].message.content,
            'total_tokens': response.usage.total_tokens,
            'duration': time.time() - start_time
        }
        self._log_api_call(log_data)
        return response

    def _parse_content(self, content):
        parsed_explanations = None
        def feature_value_to_float(value):
            # special case for S196 in ADR_PITT / Google Speech: remove some text in front of the value
            value = value.replace('Given the largely coherent narrative but with a sudden, incongruent mention of a "race horse jumping through the window," which could indicate confabulation or a breakdown in reality monitoring, I would rate this transcript as:','')
            try:
                return float(value)
            except:
                print(f"Cannot convert to float", value)
                return None

        parsed = re.findall(r"(.*?): ([0-9]+) \((.*?)\)", content)
        if len(parsed) == 0:
            # 7-step leads to 2 examples where the formatting is e.g. Word-Finding Difficulties (Anomia): 2\n- Examples from the text: ("he's has the lid off" - inco
            parsed = re.findall(r"(.*?): ([0-9]+)\n(.*?)\)", content)
        def clean_feature_name(feature_name):
            return "Word-Finding Difficulties (Anomia)" if feature_name == "Word-Finding Difficulties" else feature_name
        parsed = [(clean_feature_name(feature), value, explanation) for (feature, value, explanation) in parsed]  # wrong in one case...
        features = list(set([feature for (feature, value, explanation) in parsed]))
        parsed_values = {feature: np.mean([feature_value_to_float(value) for (f, value, explanation) in parsed if f == feature]) for feature in features}
        parsed_explanations = {feature + "_explanation": [explanation for (f, value, explanation) in parsed if f == feature] for feature in features}

        return parsed_values, parsed_explanations

    def load_gpt_features(self, sample_name, text, prompt_postfix="", seed=None):
        if seed is None:
            seed = self.seed
        system, prompt_template = self._get_prompt_templates(prompt_postfix)
        if text.strip() == "":
            print(f"No transcript provided for sample_name {sample_name}, setting None values")
            return {}, {}
        prompt = prompt_template.format(transcript=text)
        response = self._call_openai_api(sample_name, system, prompt, temperature=self.temperature, seed=seed, model=self.gpt_model)
        content = response.choices[0].message.content
        parsed_values, parsed_explanations = self._parse_content(content)
        return parsed_values, parsed_explanations






class LinguisticFeatures(Preprocessor):
    """
    Linguistic features
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Linguistic Features"
        print(f"Initializing {self.name}")

        # stanza nlp pipeline
        self.nlp_pipeline = stanza.Pipeline('en')

        # feature groups: which linguistic features to load
        valid_feature_groups = ['disfluency_features', 'literature_features', 'gpt_features', 'manual_rating_word_finding']
        try:
            self.feature_groups = self.config.config_linguistic_features.feature_groups
        except (AttributeError, KeyError):
            self.feature_groups = valid_feature_groups
        assert all([fg in valid_feature_groups for fg in self.feature_groups])

        try:
            self.gpt_feature_version = self.config.config_linguistic_features.gpt_feature_version
            if not isinstance(self.gpt_feature_version, list):
                self.gpt_feature_version = [self.gpt_feature_version]
        except (AttributeError, KeyError):
            self.gpt_feature_version = None

        try:
            self.selected_features = self.config.config_linguistic_features.selected_features
        except (AttributeError, KeyError):
            self.selected_features = []  # empty list -> select all

        if 'gpt_features' in self.feature_groups and self.gpt_feature_version is not None:
            assert len(self.gpt_feature_version) == 1, f"GPTFeatureExtractor only works for one GPT feature version for now, but got multiple: {self.gpt_feature_version}"
            try:
                gpt_model = self.config.config_linguistic_features.gpt_model
            except:
                gpt_model = None
            self.gpt_feature_extractor = GPTFeatureExtractor(self.gpt_feature_version[0], gpt_model=gpt_model, run_parameters=self.run_parameters)

        print(f"Using feature_groups {self.feature_groups}, gpt_feature_version {self.gpt_feature_version},"
              f"selected_features {self.selected_features}")


    def _count_coordinate_clauses(self, tree):
        # current node is S and has at least 2 child notes with also S => These two are coordinate clauses
        this_node_is_coordination = tree.label == 'S' and len([c for c in tree.children if c.label == 'S']) >= 2
        is_coordination = 1 if this_node_is_coordination else 0

        # add coordination here to all present in children, recursively
        return is_coordination + sum([self._count_coordinate_clauses(c) for c in tree.children])

    def _count_subordinate_clauses(self, tree):
        # current node has SBAR (subordinate clause) tagset
        is_subordination = 1 if tree.label == 'SBAR' else 0
        return is_subordination + sum([self._count_subordinate_clauses(c) for c in tree.children])

    def _count_coordinate_subordinate_clauses(self, doc):
        """
        Note that this logic is somewhat flawed, or does not really give what we want.
        Check out the test in this class for some observed problematic cases.
        Also have a look at https://github.com/jheitz/dementia/blob/main/src/scripts/coordinate_subordinate_clauses.py
        to see how it behaves on all transcripts
        """
        n_coordinate_clauses = 0
        n_subordinate_clauses = 0
        for sentence in doc.sentences:
            tree = sentence.constituency
            n_coordinate_clauses += self._count_coordinate_clauses(tree)
            n_subordinate_clauses += self._count_subordinate_clauses(tree)

        fraction = (n_coordinate_clauses + 1) / (n_subordinate_clauses + 1)
        return n_coordinate_clauses, n_subordinate_clauses, fraction

    def _calculate_disfluency_features(self, dataset: TextDataset):
        """
        Actual calculation of disfluency features is done in the ADReSSTranscriptDataLoader,
        here we just transform and summarize
        """
        assert hasattr(dataset, "disfluency_metrics"), \
            "The disfluency metrics have to be calculated during data loading for manual transcripts and should be " \
            "passed in the Dataset object. This dataset does not have it."

        disfluency_metrics_df = dataset.disfluency_metrics

        # add ratio as additional metric
        disfluency_metrics_df['n_words'] = [len(text.lower().split(" ")) for text in dataset.data]
        disfluency_metrics_df['ratio_disfluencies'] = np.where(disfluency_metrics_df['n_words'] > 0,
                                                               disfluency_metrics_df['n_disfluencies'] / disfluency_metrics_df['n_words'],
                                                               0)
        #disfluency_metrics_df = disfluency_metrics_df.drop(columns=['n_words'])

        # rename metrics to make clear it's a disfluency metric
        disfluency_metrics_df.columns = ["disfl_"+c for c in disfluency_metrics_df.columns]

        # convert to list of dicts
        features = disfluency_metrics_df.to_dict('records')
        return features

    def _calculate_literature_features_for_text(self, text):
        doc = self.nlp_pipeline(text.lower())

        feature_calculator = LinguisticFeatureLiteratureCalculator(doc, constants=self.CONSTANTS)
        count_pos, count_xpos = feature_calculator.pos_counts()
        n_words = feature_calculator.n_words()

        # todo: Use sigmoid_fraction for e.g. pronoun-noun-ratio, since it's symmetrical and actually makes the
        # features more expressive. we don't do it right now to stay consistent with prior literature.
        def sigmoid_fraction(a, b):
            assert a >= 0 and b >= 0
            a = a + 0.001 if a == 0 else a
            b = b + 0.001 if b == 0 else b
            return 1 / (1 + np.exp(-np.log(a / b)))

        features_pos = {
            'pronoun_noun_ratio': safe_divide(count_pos['PRON'], count_pos['NOUN']),  # [1], [2], [4]
            'verb_noun_ratio': safe_divide(count_pos['VERB'], count_pos['NOUN']),  # [4]
            'subordinate_coordinate_conjunction_ratio': safe_divide(count_pos['SCONJ'], count_pos['CCONJ']),  # [3]
            'adverb_ratio': safe_divide(count_pos['ADV'], n_words),  # [1], [2], [9]
            'noun_ratio': safe_divide(count_pos['NOUN'], n_words),  # [1], [8], [9]
            'verb_ratio': safe_divide(count_pos['VERB'], n_words),  # [1], [9]
            'pronoun_ratio': safe_divide(count_pos['PRON'], n_words),  # [2], [9]
            'personal_pronoun_ratio': safe_divide(count_xpos['PRP'], n_words),  # [2]
            'determiner_ratio': safe_divide(count_pos['DET'], n_words),  # [8]
            'preposition_ratio': safe_divide(count_xpos['IN'], n_words),  # [9]
            'verb_present_participle_ratio': safe_divide(count_xpos['VBG'], n_words),  # [2, 8]
            'verb_modal_ratio': safe_divide(count_xpos['MD'], n_words),  # [8]
            'verb_third_person_singular_ratio': safe_divide(count_xpos['VBZ'], n_words),
            # [1] (I suppose by inflected verbs they mean 3. person)
        }

        constituency_rules_count = feature_calculator.constituency_rules_count()

        constituents = feature_calculator.get_constituents()
        NP = [c for c in constituents if c['label'] == 'NP']
        PP = [c for c in constituents if c['label'] == 'PP']
        VP = [c for c in constituents if c['label'] == 'VP']
        PRP = [c for c in constituents if c['label'] == 'PRP']

        features_constituency = {
            # NP -> PRP means "count the number of noun phrases (NP) that consist of a pronoun (PRP)"
            'NP -> PRP': constituency_rules_count['NP -> PRP'],  # [1]
            'ADVP -> RB': constituency_rules_count['ADVP -> RB'],  # [1], [2]
            'NP -> DT_NN': constituency_rules_count['NP -> DT_NN'],  # [1]
            'ROOT -> FRAG': constituency_rules_count['ROOT -> FRAG'],  # [1]
            'VP -> AUX_VP': constituency_rules_count['VP -> AUX_VP'],  # [1]
            'VP -> VBG': constituency_rules_count['VP -> VBG'],  # [1]
            'VP -> VBG_PP': constituency_rules_count['VP -> VBG_PP'],  # [1]
            'VP -> IN_S': constituency_rules_count['VP -> IN_S'],  # [1]
            'VP -> AUX_ADJP': constituency_rules_count['VP -> AUX_ADJP'],  # [1]
            'VP -> AUX': constituency_rules_count['VP -> AUX'],  # [1]
            'VP -> VBD_NP': constituency_rules_count['VP -> VBD_NP'],  # [1]
            'INTJ -> UH': constituency_rules_count['INTJ -> UH'],  # [1]
            'NP_ratio': safe_divide(len(NP), len(constituents)),  # [9]
            'PRP_ratio': safe_divide(len(PRP), len(constituents)),  # [9]
            'PP_ratio': safe_divide(len(PP), len(constituents)),  # [1]
            'VP_ratio': safe_divide(len(VP), len(constituents)),  # [1]
            'avg_n_words_in_NP': safe_divide(sum([len(c['text'].split(" ")) for c in NP]), len(NP)),  # [9]
        }

        simple_features = {
            'n_words': n_words,  # [9], [4], [6], [8]
            'n_unique_words': feature_calculator.n_unique(),  # [6], [8]
            'avg_word_length': feature_calculator.word_length(),  # [1], [2]
            'avg_sentence_length': feature_calculator.sentence_length(),  # [4]
            'words_not_in_dict_ratio': feature_calculator.not_in_dictionary(),  # [1], [2]
        }

        vocabulary_richness_features = {
            'brunets_index': feature_calculator.brunets_indes(),  # [3], [8]
            'honores_statistic': feature_calculator.honores_statistic(),  # [1], [9], [3], [8]
            'ttr': feature_calculator.ttr(),  # [4], [8]
            'mattr': feature_calculator.mattr(),  # [8]
        }

        readability_features = {
            'flesch_kincaid': feature_calculator.flesch_kincaid()  # [3]
        }

        avg_distance_between_utterances, prop_dist_thresh_05 = feature_calculator.cos_dist_between_sentences()
        repetitiveness_features = {
            'avg_distance_between_utterances': avg_distance_between_utterances,  # [1], [2]
            'prop_utterance_dist_below_05': prop_dist_thresh_05  # [1], [2]
        }

        density_features = {
            'propositional_density': safe_divide(
                count_pos['VERB'] + count_pos['ADJ'] + count_pos['ADV'] + count_pos['ADP'] +
                count_pos['CCONJ'] + count_pos['SCONJ'], n_words),  # [3], [7]
            'content_density': safe_divide(
                count_pos['NOUN'] + count_pos['VERB'] + count_pos['ADJ'] + count_pos['ADV'], n_words),
            # [3], [8], [9]
        }

        all_features = {**features_pos, **features_constituency, **simple_features, **vocabulary_richness_features,
                        **readability_features, **repetitiveness_features, **density_features}

        # let's drop the null feature (0 in every sample in ADReSS):
        all_features = {f: all_features[f] for f in all_features if
                        f not in ['VP -> AUX_VP', 'VP -> IN_S', 'VP -> AUX_ADJP', 'VP -> AUX']}

        # rename features
        all_features = {f"lit_{f}": all_features[f] for f in all_features}


        return all_features

    def _load_literature_features(self, dataset):
        features = [self._calculate_literature_features_for_text(text) for text in dataset.data]
        return features

    def _load_gpt_features(self, dataset):
        features = [self.gpt_feature_extractor.load_gpt_features(sample_name, text) for sample_name, text in zip(dataset.sample_names, dataset.data)]
        feature_values, explanations = [values for values, _ in features], [explanations for _, explanations in features]
        values_df = pd.DataFrame(feature_values)
        values_df['sample_name'] = dataset.sample_names
        explanations_df = pd.DataFrame(explanations)
        explanations_df['sample_name'] = dataset.sample_names
        store_dir = os.path.join(self.CONSTANTS.PREPROCESSED_DATA, "gpt_features",
                                 self.gpt_feature_extractor.version, dataset_name_to_url_part(dataset.name))
        create_directory(store_dir)
        values_df.to_csv(os.path.join(store_dir, "values.csv"), index=False)
        if any([expl is not None for expl in explanations]):
            explanations_df.to_csv(os.path.join(store_dir, "explanations.csv"), index=False)

        return feature_values

    def _load_manual_rating_word_finding(self, dataset):
        """
        Word finding difficulty according to ``Rating Scale Profile of Speech Characteristics" of the Boston Diagnostic Aphasia Examination (BDAE)
        The ratings have been collected using a custom WebApp
        """
        features_df = pd.read_csv(os.path.join(self.CONSTANTS.RESOURCES_DIR, "manual_ratings_word_finding_processed.csv"))
        features_df = features_df.set_index("sample_name")
        features_df.column = [f"WordFinding_{rater_id}" for rater_id in features_df.columns]
        features = [features_df.loc[sample_name].to_dict() for sample_name in dataset.sample_names]

        return features


    def _load_features(self, dataset: TextDataset):
        assert isinstance(dataset, TextDataset), "Input should be TextDataset (manual transcripts)"
        if 'transcript_config' in dataset.config:
            # manual transcripts, check that it's the right version
            transcript_config = dataset.config['transcript_config']
            transcript_config_hash = dataset.config['transcript_config_hash']

            text_has_terminators = transcript_config['keep_terminators'] or 'Automatic Punctuation' in dataset.config[
                'preprocessors']
            assert text_has_terminators, "If there are no terminators (periods), some of the linguistic metrics don't make sense"
            assert not transcript_config[
                'keep_pauses'], "Don't use linguistic features if there are explicit pauses, this might mess it up"
            # assert transcript_config['only_PAR'], "Don't calculate linguistic features on interviewer"
            if not transcript_config['only_PAR']:
                print(
                    "\n\n\n WARNING: You're calculating linguistic features on interviewer speech, not just participant. Is this on purpose?\n\n")

        else:
            # should be automatic transcripts, make sure there really was an ASR involved
            assert any(['ASR' in p for p in self.config.preprocessors])

        features_collected = []
        for feature_group in self.feature_groups:
            if feature_group == 'disfluency_features':
                features_collected.append(self._calculate_disfluency_features(dataset))
            elif feature_group == 'literature_features':
                features_collected.append(self._load_literature_features(dataset))
            elif feature_group == 'gpt_features':
                features_collected.append(self._load_gpt_features(dataset))
            elif feature_group == 'manual_rating_word_finding':
                features_collected.append(self._load_manual_rating_word_finding(dataset))
            else:
                raise ValueError(f"Invalid feature group {feature_group}")

        # combine multiple (feature group) dictionaries of {feature_name: value} for each sample,
        # getting one dictionary per sample
        all_features = [reduce(lambda a, b: {**a, **b}, sample) for sample in zip(*features_collected)]

        # select specific subset of features
        if len(self.selected_features) > 0:
            non_existing = [f for f in self.selected_features if f not in all_features[0].keys()]
            if len(non_existing) > 0:
                print("Warning: the following features requested in the config file have do not exist:", non_existing)
            all_features = [{feature: sample[feature] for feature in sample if feature in self.selected_features} for sample in all_features]

        features_df = pd.DataFrame(all_features)

        # dropping nan rows
        non_nan_filter = ~features_df.isna().any(axis=1)
        sample_names = np.array(dataset.sample_names)[non_nan_filter]
        nan_row_sample_names = np.array(dataset.sample_names)[~non_nan_filter]
        labels = np.array(dataset.labels)[non_nan_filter]
        n_rows_before = features_df.shape[0]
        features_df = features_df.dropna()
        if np.sum(~non_nan_filter) > 0:
            print(f"\n\nATTENTION: null values in {np.sum(~non_nan_filter)} rows / {n_rows_before} (sample names {nan_row_sample_names})\n\n")

        config_without_preprocessors = {key: dataset.config[key] for key in dataset.config if key != 'preprocessors'}
        new_config = {
            'preprocessors': [*dataset.config['preprocessors'], self.name],
            **config_without_preprocessors
        }
        return TabularDataset(data=features_df, labels=labels, sample_names=sample_names,
                              name=f"{dataset.name} - {self.name}", config=new_config)



    def preprocess_dataset(self, dataset: TextDataset) -> TabularDataset:
        print(f"Calculating linguistic features for dataset {dataset}")
        return self._load_features(dataset)

