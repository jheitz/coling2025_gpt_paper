datetime_str=$(date '+%Y%m%d_%H%M')_$(openssl rand -hex 2)

### Collect results for GPT paper - uncomment respective files

## 10-fold Cross Validation
#python -u run.py --config ../configs/linguistic_features_different_sets/cross_validation/manual/linguistic_features_gpt_5features.yaml --runname manual_linguistic_features_gpt_5features --results_base_dir ${datetime_str}_feature_sets_cv
#python -u run.py --config ../configs/linguistic_features_different_sets/cross_validation/manual/linguistic_features_gpt_5features_literature.yaml --runname manual_linguistic_features_gpt_5features_literature --results_base_dir ${datetime_str}_feature_sets_cv
#python -u run.py --config ../configs/linguistic_features_different_sets/cross_validation/manual/linguistic_features_gpt4o_5features.yaml --runname manual_linguistic_features_gpt4o_5features --results_base_dir ${datetime_str}_feature_sets_cv
#python -u run.py --config ../configs/linguistic_features_different_sets/cross_validation/manual/linguistic_features_gpt4o_5features_literature.yaml --runname manual_linguistic_features_gpt4o_5features_literature --results_base_dir ${datetime_str}_feature_sets_cv
#python -u run.py --config ../configs/linguistic_features_different_sets/cross_validation/manual/linguistic_features_literature.yaml --runname manual_linguistic_features_literature --results_base_dir ${datetime_str}_feature_sets_cv
#python -u run.py --config ../configs/linguistic_features_different_sets/cross_validation/google/linguistic_features_gpt_5features.yaml --runname google_linguistic_features_gpt_5features --results_base_dir ${datetime_str}_feature_sets_cv
#python -u run.py --config ../configs/linguistic_features_different_sets/cross_validation/google/linguistic_features_gpt_5features_literature.yaml --runname google_linguistic_features_gpt_5features_literature --results_base_dir ${datetime_str}_feature_sets_cv
python -u run.py --config ../configs/linguistic_features_different_sets/cross_validation/google/linguistic_features_gpt4o_5features.yaml --runname google_linguistic_features_gpt4o_5features --results_base_dir ${datetime_str}_feature_sets_cv
#python -u run.py --config ../configs/linguistic_features_different_sets/cross_validation/google/linguistic_features_gpt4o_5features_literature.yaml --runname google_linguistic_features_gpt4o_5features_literature --results_base_dir ${datetime_str}_feature_sets_cv
#python -u run.py --config ../configs/linguistic_features_different_sets/cross_validation/google/linguistic_features_literature.yaml --runname google_linguistic_features_literature --results_base_dir ${datetime_str}_feature_sets_cv
#python -u run.py --config ../configs/linguistic_features_different_sets/cross_validation/whisper/linguistic_features_gpt_5features.yaml --runname whisper_linguistic_features_gpt_5features --results_base_dir ${datetime_str}_feature_sets_cv
#python -u run.py --config ../configs/linguistic_features_different_sets/cross_validation/whisper/linguistic_features_gpt_5features_literature.yaml --runname whisper_linguistic_features_gpt_5features_literature --results_base_dir ${datetime_str}_feature_sets_cv
#python -u run.py --config ../configs/linguistic_features_different_sets/cross_validation/whisper/linguistic_features_gpt4o_5features.yaml --runname whisper_linguistic_features_gpt4o_5features --results_base_dir ${datetime_str}_feature_sets_cv
#python -u run.py --config ../configs/linguistic_features_different_sets/cross_validation/whisper/linguistic_features_gpt4o_5features_literature.yaml --runname whisper_linguistic_features_gpt4o_5features_literature --results_base_dir ${datetime_str}_feature_sets_cv
#python -u run.py --config ../configs/linguistic_features_different_sets/cross_validation/whisper/linguistic_features_literature.yaml --runname whisper_linguistic_features_literature --results_base_dir ${datetime_str}_feature_sets_cv

## Train on ADReSS train set, test on ADReSS test set - results not included in final publication
#python -u run.py --config ../configs/linguistic_features_different_sets/train_test/manual/linguistic_features_gpt_5features.yaml --runname manual_linguistic_features_gpt_5features --results_base_dir ${datetime_str}_feature_sets_traintest
#python -u run.py --config ../configs/linguistic_features_different_sets/train_test/manual/linguistic_features_gpt_5features_literature.yaml --runname manual_linguistic_features_gpt_5features_literature --results_base_dir ${datetime_str}_feature_sets_traintest
#python -u run.py --config ../configs/linguistic_features_different_sets/train_test/manual/linguistic_features_gpt4o_5features.yaml --runname manual_linguistic_features_gpt4o_5features --results_base_dir ${datetime_str}_feature_sets_traintest
#python -u run.py --config ../configs/linguistic_features_different_sets/train_test/manual/linguistic_features_gpt4o_5features_literature.yaml --runname manual_linguistic_features_gpt4o_5features_literature --results_base_dir ${datetime_str}_feature_sets_traintest
#python -u run.py --config ../configs/linguistic_features_different_sets/train_test/manual/linguistic_features_literature.yaml --runname manual_linguistic_features_literature --results_base_dir ${datetime_str}_feature_sets_traintest
#python -u run.py --config ../configs/linguistic_features_different_sets/train_test/google/linguistic_features_gpt_5features.yaml --runname google_linguistic_features_gpt_5features --results_base_dir ${datetime_str}_feature_sets_traintest
#python -u run.py --config ../configs/linguistic_features_different_sets/train_test/google/linguistic_features_gpt_5features_literature.yaml --runname google_linguistic_features_gpt_5features_literature --results_base_dir ${datetime_str}_feature_sets_traintest
#python -u run.py --config ../configs/linguistic_features_different_sets/train_test/google/linguistic_features_gpt4o_5features.yaml --runname google_linguistic_features_gpt4o_5features --results_base_dir ${datetime_str}_feature_sets_traintest
#python -u run.py --config ../configs/linguistic_features_different_sets/train_test/google/linguistic_features_gpt4o_5features_literature.yaml --runname google_linguistic_features_gpt4o_5features_literature --results_base_dir ${datetime_str}_feature_sets_traintest
#python -u run.py --config ../configs/linguistic_features_different_sets/train_test/google/linguistic_features_literature.yaml --runname google_linguistic_features_literature --results_base_dir ${datetime_str}_feature_sets_traintest
#python -u run.py --config ../configs/linguistic_features_different_sets/train_test/whisper/linguistic_features_gpt_5features.yaml --runname whisper_linguistic_features_gpt_5features --results_base_dir ${datetime_str}_feature_sets_traintest
#python -u run.py --config ../configs/linguistic_features_different_sets/train_test/whisper/linguistic_features_gpt_5features_literature.yaml --runname whisper_linguistic_features_gpt_5features_literature --results_base_dir ${datetime_str}_feature_sets_traintest
#python -u run.py --config ../configs/linguistic_features_different_sets/train_test/whisper/linguistic_features_gpt4o_5features.yaml --runname whisper_linguistic_features_gpt4o_5features --results_base_dir ${datetime_str}_feature_sets_traintest
#python -u run.py --config ../configs/linguistic_features_different_sets/train_test/whisper/linguistic_features_gpt4o_5features_literature.yaml --runname whisper_linguistic_features_gpt4o_5features_literature --results_base_dir ${datetime_str}_feature_sets_traintest
#python -u run.py --config ../configs/linguistic_features_different_sets/train_test/whisper/linguistic_features_literature.yaml --runname whisper_linguistic_features_literature --results_base_dir ${datetime_str}_feature_sets_traintest

## GPT 3 fine-tuning - results not included in final paper - this was for a first version of this paper, at a time when GPT-4o was not available yet
#python -u run.py --config ../configs/gpt-finetuned/gpt-finetuned-google-cv.yaml --runname gpt-finetuned-google-cv --results_base_dir ${datetime_str}_results_for_paper_gpt_finetuned
#python -u run.py --config ../configs/gpt-finetuned/gpt-finetuned-google-test_on_testset.yaml --runname gpt-finetuned-google-test_on_testset --results_base_dir ${datetime_str}_results_for_paper_gpt_finetuned
#python -u run.py --config ../configs/gpt-finetuned/gpt-finetuned-manual-cv.yaml --runname gpt-finetuned-manual-cv --results_base_dir ${datetime_str}_results_for_paper_gpt_finetuned
#python -u run.py --config ../configs/gpt-finetuned/gpt-finetuned-manual-test_on_testset.yaml --runname gpt-finetuned-manual-test_on_testset --results_base_dir ${datetime_str}_results_for_paper_gpt_finetuned
#python -u run.py --config ../configs/gpt-finetuned/gpt-finetuned-whisper-cv.yaml --runname gpt-finetuned-whisper-cv --results_base_dir ${datetime_str}_results_for_paper_gpt_finetuned
#python -u run.py --config ../configs/gpt-finetuned/gpt-finetuned-whisper-test_on_testset.yaml --runname gpt-finetuned-whisper-test_on_testset --results_base_dir ${datetime_str}_results_for_paper_gpt_finetuned

## GPT 4o fine-tuning
#python -u run.py --config ../configs/gpt-finetuned/gpt-4o-finetuned-google-cv.yaml --runname gpt-4o-finetuned-google-cv --results_base_dir ${datetime_str}_results_for_paper_gpt_finetuned
#python -u run.py --config ../configs/gpt-finetuned/gpt-4o-finetuned-google-test_on_testset.yaml --runname gpt-4o-finetuned-google-test_on_testset --results_base_dir ${datetime_str}_results_for_paper_gpt_finetuned
#python -u run.py --config ../configs/gpt-finetuned/gpt-4o-finetuned-manual-cv.yaml --runname gpt-4o-finetuned-manual-cv --results_base_dir ${datetime_str}_results_for_paper_gpt_finetuned
#python -u run.py --config ../configs/gpt-finetuned/gpt-4o-finetuned-manual-test_on_testset.yaml --runname gpt-4o-finetuned-manual-test_on_testset --results_base_dir ${datetime_str}_results_for_paper_gpt_finetuned
#python -u run.py --config ../configs/gpt-finetuned/gpt-4o-finetuned-whisper-cv.yaml --runname gpt-4o-finetuned-whisper-cv --results_base_dir ${datetime_str}_results_for_paper_gpt_finetuned
#python -u run.py --config ../configs/gpt-finetuned/gpt-4o-finetuned-whisper-test_on_testset.yaml --runname gpt-4o-finetuned-whisper-test_on_testset --results_base_dir ${datetime_str}_results_for_paper_gpt_finetuned

## Extra results: GPT zero shot prediction
#python -u run.py --config ../configs/gpt-zeroshot/gpt-4o-zeroshot-google-cv.yaml --runname gpt-4o-zeroshot-google-cv --results_base_dir ${datetime_str}_results_for_paper_gpt_zeroshot
#python -u run.py --config ../configs/gpt-zeroshot/gpt-4o-zeroshot-manual-cv.yaml --runname gpt-4o-zeroshot-manual-cv --results_base_dir ${datetime_str}_results_for_paper_gpt_zeroshot
#python -u run.py --config ../configs/gpt-zeroshot/gpt-4o-zeroshot-whisper-cv.yaml --runname gpt-4o-zeroshot-whisper-cv --results_base_dir ${datetime_str}_results_for_paper_gpt_zeroshot
#python -u run.py --config ../configs/gpt-zeroshot/gpt-zeroshot-google-cv.yaml --runname gpt-zeroshot-google-cv --results_base_dir ${datetime_str}_results_for_paper_gpt_zeroshot
#python -u run.py --config ../configs/gpt-zeroshot/gpt-zeroshot-manual-cv.yaml --runname gpt-zeroshot-manual-cv --results_base_dir ${datetime_str}_results_for_paper_gpt_zeroshot
#python -u run.py --config ../configs/gpt-zeroshot/gpt-zeroshot-whisper-cv.yaml --runname gpt-zeroshot-whisper-cv --results_base_dir ${datetime_str}_results_for_paper_gpt_zeroshot

## Analysis correlation of Word-Finding difficulties against disfluency_ratio vs. other linguistic features
#python -u run.py --config ../configs/validation_against_proxy_measure/literature_gpt_and_disfluency_features.yaml --runname literature_gpt_and_disfluency_features

## Extra analysis - check results for 10 GPT feature (instead of 5)
#python -u run.py --config ../configs/linguistic_features_different_sets/cross_validation/manual/linguistic_features_gpt_10features.yaml --runname linguistic_features_gpt_10features --results_base_dir ${datetime_str}_feature_sets_cv
#python -u run.py --config ../configs/linguistic_features_different_sets/cross_validation/manual/linguistic_features_gpt_10features_literature.yaml --runname linguistic_features_gpt_10features_literature --results_base_dir ${datetime_str}_feature_sets_cv

