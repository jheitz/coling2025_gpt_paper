import shap
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import calibration_curve
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from model.base_model import BaseModel
from dataloader.dataset import TabularDataset
from util.helpers import python_to_json, plot_roc, plot_roc_cv, store_obj_to_disk
from dataloader.custom_data_splitter import ADReSSTrainValTestSplitter
from evaluation.bias_analysis import BiasAnalysis

class TreeBasedClassifier(BaseModel):
    """
    Tree-based classification (Random Forest, GradientBoosting) for tabular data
    """
    def __init__(self, model_name, *args, **kwargs):
        super().__init__(model_name, *args, **kwargs)
        self.model = None

        # specificity is the recall of the negative class
        specificity = lambda y_true, y_pred: metrics.recall_score(y_true, y_pred, pos_label=0)
        self.binary_metrics = {
            'accuracy': metrics.accuracy_score,
            'f1': metrics.f1_score,
            'precision': metrics.precision_score,
            'recall': metrics.recall_score,
            'specificity': specificity,
            'confusion_matrix': metrics.confusion_matrix
        }
        self.continous_metrics = {
            'roc_auc': metrics.roc_auc_score,
            #'roc_curve': metrics.roc_curve
        }

        ## Some configuration

        # Use cross validation if cv_splits > 1. If cv_splits == 1, use predefined train and test sets
        try:
            self.cv_splits = self.config.cv_splits
        except AttributeError:
            self.cv_splits = 10

        # whether or not to store the model to disk for future analysis
        try:
            self.store_model = self.config.store_model
        except AttributeError:
            self.store_model = False

        try:
            self.n_estimators = self.config.config_model.n_estimators
        except (AttributeError, KeyError):
            self.n_estimators = 500

        try:
            self.max_depth = self.config.config_model.max_depth
        except (AttributeError, KeyError):
            self.max_depth = None

        try:
            self.min_samples_leaf = self.config.config_model.min_samples_leaf
        except (AttributeError, KeyError):
            self.min_samples_leaf = 1

        # Learning rate, for gradient boosting
        try:
            self.learning_rate = self.config.config_model.learning_rate
        except (AttributeError, KeyError):
            self.learning_rate = 0.1

        print(f"Using store_model {self.store_model}, cv_splits {self.cv_splits}, n_estimators {self.n_estimators}, "
              f"learning_rate {self.learning_rate}, max_depth {self.max_depth}, "
              f"min_samples_leaf {self.min_samples_leaf}")


    def _save_model(self, split_idx=None):
        # store model to disk so we can later explore it
        # only if store_model = True and for the first split
        if self.store_model and (split_idx == 1 or split_idx is None):
            raise NotImplementedError()

    def set_train(self, dataset: TabularDataset):
        self._train = dataset

    def set_test(self, dataset: TabularDataset):
        self._test = dataset

    def prepare_data(self):
        pass

    def train(self, train_set=None, split_idx=None):
        """
        One training using train_set, if given, or self._train otherwise

        :param train_set: train set of this CV split
        :param split_idx: CV split Number
        :return:
        """
        assert train_set is not None

        if self.name == 'RandomForest':
            self.model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                                min_samples_leaf=self.min_samples_leaf)
        elif self.name == 'GradientBoosting':
            self.model = GradientBoostingClassifier(n_estimators=self.n_estimators, learning_rate=self.learning_rate)
        else:
            raise ValueError("Invalid model for tree-based classifier, can only be RandomForest or GradientBoosting")

        # Print some identifiers of items in sets
        # This is to be able to manually check that the splits are identical over different runs
        def subset_identifer(data_set):
            # get the sum of the tensor elements of the first and last sample in each split
            sums = [np.sum(row) for row in data_set.data.values]
            return [sums[0], sums[-1]]
        print(f"Train set: {subset_identifer(train_set)}...")
        print(f"Label counts")
        print(pd.Series(train_set.labels).value_counts())

        if split_idx == 0 or split_idx is None:
            print("Model used:")
            print(self.model)

        print(f"Start training", f"split {split_idx}..." if split_idx is not None else "")

        self.model.fit(train_set.data, train_set.labels)

        # test on train set -> training error
        predictions = self.model.predict_proba(train_set.data)[:, 1]
        computed_metrics_binary = {name: self.binary_metrics[name](train_set.labels, np.round(predictions).astype(int))
                                   for name in self.binary_metrics}
        computed_metrics_continuous = {name: self.continous_metrics[name](train_set.labels, predictions)
                                       for name in self.continous_metrics}
        computed_metrics = {**computed_metrics_binary, **computed_metrics_continuous}
        computed_metrics['predictions'] = predictions
        computed_metrics['labels'] = train_set.labels
        return computed_metrics


    def test(self, test_set=None, split_idx=None):
        """
        Test self.model on test_set, if given (for cross-validation), or self._test, if not
        """
        assert test_set is not None
        assert self.model is not None
        print("Start testing", f"split {split_idx}..." if split_idx is not None else "")

        predictions = self.model.predict_proba(test_set.data)[:,1]

        computed_metrics_binary = {name: self.binary_metrics[name](test_set.labels, np.round(predictions).astype(int))
                                   for name in self.binary_metrics}
        computed_metrics_continuous = {name: self.continous_metrics[name](test_set.labels, predictions)
                                       for name in self.continous_metrics}
        computed_metrics = {**computed_metrics_binary, **computed_metrics_continuous}
        computed_metrics['predictions'] = predictions
        computed_metrics['labels'] = test_set.labels
        computed_metrics['sample_names'] = test_set.sample_names

        # sklearn feature importance
        # for both RandomForestClassifier and GradientBoostingClassifier, this is
        # "The impurity-based feature importances, (...) also known as the Gini importance."
        computed_metrics['gini_feature_importance'] = self.model.feature_importances_

        # shap value explainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(test_set.data)
        if isinstance(shap_values, list):
            # There are two outputs, for each output class, happens for RandomForest
            assert len(shap_values) == 2, f"Expected one shap_value output per class, but it's {len(shap_values)}"
            shap_values = shap_values[1]
        else:
            # Only one output, for GradientBoosting
            pass
        assert isinstance(shap_values, np.ndarray), "Shap_value output should be ndarray"
        computed_metrics['shap_values'] = shap_values

        return computed_metrics

    def train_test(self):
        test_metrics = {}
        train_metrics = {}
        all_predictions, all_labels = [], []  # for ROC curves and other further analysis
        all_sample_names = [] # for further analysis
        all_test_data, all_shap_values = [], []  # for summary plots over all splits
        all_gini_feature_importances = []

        complete_dataset = self._train.concatenate(self._test)

        # store dataset to disk, to do analysis on features
        complete_dataset.store_to_disk(os.path.join(self.run_parameters.results_dir, "dataset"))

        if self.cv_splits > 1:
            # We use cross validation, i.e. we first combine the train and test set and then use the pre-defined
            # (deterministic) mapping to split it
            complete_dataset.load_cv_split_assignment(self.cv_splits)

            #kfold = StratifiedKFold(n_splits=self.cv_splits)
            #for split_idx, (train_indices, test_indices) in enumerate(kfold.split(X=np.zeros(len(complete_dataset.labels)), y=complete_dataset.labels)):

            for split_idx in range(self.cv_splits):
                test_indices = np.where(complete_dataset.cv_test_splits == split_idx)[0]
                train_indices = np.where(complete_dataset.cv_test_splits != split_idx)[0]
                train_set = complete_dataset.subset_from_indices(train_indices)
                test_set = complete_dataset.subset_from_indices(test_indices)

                train_results = self.train(train_set, split_idx)
                train_metrics[f'split_{split_idx}'] = train_results

                test_results = self.test(test_set, split_idx=split_idx)
                test_metrics[f'split_{split_idx}'] = test_results
                all_test_data.append(test_set.data)

                self._save_model(split_idx)


        elif self.cv_splits == -1:
            # Train on part of train, validate on validation set (in /resources).
            # This is used to compare to some GPT experiment I'm doing, where I don't want to touch the
            # test set yet
            data_splitter = ADReSSTrainValTestSplitter(self.CONSTANTS)
            dataset = self._train.concatenate(self._test)
            test_indices = np.where(
                np.isin(dataset.sample_names, data_splitter.get_mapping().query("split == 'Validation'").sample_name))[0]
            train_indices = np.where(
                np.isin(dataset.sample_names, data_splitter.get_mapping().query("split == 'Train'").sample_name))[0]
            train_set = dataset.subset_from_indices(train_indices)
            test_set = dataset.subset_from_indices(test_indices)
            train_metrics[f'train'] = self.train(train_set)
            test_results = self.test(test_set)
            test_metrics[f'test'] = test_results
            all_test_data.append(test_set.data)
            self._save_model()

        else:
            # No cross validation, use existing train / test split
            train_set = self._train
            test_set = self._test
            train_metrics[f'train'] = self.train(train_set)
            test_results = self.test(test_set)
            test_metrics['test'] = test_results
            all_test_data.append(test_set.data)
            self._save_model()


        print("Test metrics:", test_metrics)

        all_predictions = [test_metrics[key]['predictions'] for key in test_metrics]
        all_labels = [test_metrics[key]['labels'] for key in test_metrics]
        all_sample_names = [test_metrics[key]['sample_names'] for key in test_metrics]
        all_shap_values = [test_metrics[key]['shap_values'] for key in test_metrics]
        all_gini_feature_importances = [test_metrics[key]['gini_feature_importance'] for key in test_metrics]

        if len(test_metrics.keys()) > 1: # cv -> multiple test results
            all_predictions_flat = np.concatenate(all_predictions)
            all_labels_flat = np.concatenate(all_labels)
            computed_metrics_binary = {
                name: self.binary_metrics[name](all_labels_flat, np.round(all_predictions_flat).astype(int))
                for name in self.binary_metrics}
            computed_metrics_continuous = {name: self.continous_metrics[name](all_labels_flat, all_predictions_flat)
                                           for name in self.continous_metrics}
            computed_metrics = {**computed_metrics_binary, **computed_metrics_continuous}
            print("CV test metrics aggregated", computed_metrics)

        # write metrics to file (as json)
        with open(os.path.join(self.run_parameters.results_dir, "metrics.txt"), "w") as file:
            file.write(python_to_json(test_metrics))

        # overall train and test performance (roc_auc)
        train_error = metrics.roc_auc_score(
            [l for split in train_metrics for l in train_metrics[split]['labels']],
            [l for split in train_metrics for l in train_metrics[split]['predictions']],
        )
        test_error = metrics.roc_auc_score(
            [l for split in test_metrics for l in test_metrics[split]['labels']],
            [l for split in test_metrics for l in test_metrics[split]['predictions']],
        )
        print("Overall train error:", train_error)
        print("Overall test error:", test_error)
        with open(os.path.join(self.run_parameters.results_dir, "train_test_performance.txt"), "w") as file:
            file.write(python_to_json({"train_roc_auc": train_error, 'test_roc_auc': test_error}))

        # write test predictions, labels, sample_names to file
        with open(os.path.join(self.run_parameters.results_dir, "predictions.txt"), "w") as file:
            file.write(python_to_json(all_predictions))
        with open(os.path.join(self.run_parameters.results_dir, "labels.txt"), "w") as file:
            file.write(python_to_json(all_labels))
        with open(os.path.join(self.run_parameters.results_dir, "sample_names.txt"), "w") as file:
            file.write(python_to_json(all_sample_names))

        # write roc curve to results dir
        plot_roc(os.path.join(self.run_parameters.results_dir, "roc.png"), all_predictions, all_labels)
        plot_roc_cv(os.path.join(self.run_parameters.results_dir, "roc_cv.png"), all_predictions, all_labels)

        # write gini feature importance to file
        self._write_gini_feature_importance(list(complete_dataset.data.columns), all_gini_feature_importances)

        # write feature correlation matrix to file
        self._write_correlation_matrix(complete_dataset.data)

        # write feature distributions per file to disk
        self._write_feature_distribution(complete_dataset)

        # plot calibraiton
        self._plot_calibration(all_labels, all_predictions)

        # write shap values to file
        all_shap_values_flat = np.concatenate(all_shap_values)
        all_test_data_flat = pd.concat(all_test_data)
        shap.summary_plot(all_shap_values_flat, all_test_data_flat, show=False)
        plt.savefig(os.path.join(self.run_parameters.results_dir, "shap.png"))
        store_obj_to_disk("shap_values", all_shap_values_flat, self.run_parameters.results_dir)

        # run bias analysis
        bias_analysis = BiasAnalysis(constants=self.CONSTANTS, experiment_name=self.run_parameters.runname)
        bias_analysis.bias_analysis(all_predictions, all_labels, all_sample_names,
                                    os.path.join(self.run_parameters.results_dir, "bias.png"))

        print("Done.")

    def _write_gini_feature_importance(self, features, all_gini_feature_importances):
        features = np.array(features)
        importances = np.mean(all_gini_feature_importances, axis=0)  # mean over splits
        indices = np.argsort(importances)

        with open(os.path.join(self.run_parameters.results_dir, "gini_feature_importance.txt"), "w") as file:
            file.write(python_to_json(list(zip(features, importances))))

        plt.figure(figsize=(6, len(features)/4))
        plt.title('Gini feature Importances')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), features[indices])
        plt.xlabel('Relative Importance (gini)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_parameters.results_dir, "gini_feature_importance.png"))
        plt.close()

    def _write_correlation_matrix(self, data: pd.DataFrame):
        n_features = data.shape[1]
        f = plt.figure(figsize=(n_features*.3+4, n_features*.3))
        correlation_matrix = data.corr()
        plt.matshow(correlation_matrix, fignum=f.number, cmap=plt.get_cmap("bwr"), vmin=-1, vmax=1)
        plt.xticks(range(data.select_dtypes(['number']).shape[1]), data.select_dtypes(['number']).columns, fontsize=14,
                   rotation=45, ha='left', rotation_mode='anchor')
        plt.yticks(range(data.select_dtypes(['number']).shape[1]), data.select_dtypes(['number']).columns, fontsize=14)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        plt.title('Correlation Matrix', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_parameters.results_dir, "feature_correlation.png"))
        plt.close()

        with open(os.path.join(self.run_parameters.results_dir, "feature_correlation.txt"), "w") as file:
            file.write(python_to_json(correlation_matrix))

    def _plot_calibration(self, all_labels, all_predictions):
        all_predictions_flat = np.concatenate(all_predictions)
        all_labels_flat = np.concatenate(all_labels)

        true_pos, pred_pos = calibration_curve(all_labels_flat, all_predictions_flat, n_bins=10)

        # Plot the Probabilities Calibrated curve
        plt.plot(pred_pos,
                 true_pos,
                 marker='o',
                 linewidth=1,
                 label='Classifier')

        # Plot the Perfectly Calibrated by Adding the 45-degree line to the plot
        plt.plot([0, 1],
                 [0, 1],
                 linestyle='--',
                 label='Perfectly Calibrated')

        # Set the title and axis labels for the plot
        plt.title('Probability Calibration Curve')
        plt.xlabel('Predicted Probability')
        plt.ylabel('True Probability')

        # Add a legend to the plot
        plt.legend(loc='best')

        plt.tight_layout()
        plt.savefig(os.path.join(self.run_parameters.results_dir, "calibration.png"))
        plt.close()

    def _store_dataset(self, complete_dataset):
        complete_dataset.data.to_pickle("dataset_features.pkl")
        complete_dataset.save("dataset_labels")

    def _write_feature_distribution(self, dataset):
        def plot_distribution(features_dementia, features_control, all_features):
            feature_names = all_features.columns
            positions_dementia = [i * 3 for i in range(len(feature_names))]
            positions_control = [i * 3 + 1 for i in range(len(feature_names))]
            tick_positions = [i * 3 + 0.5 for i in range(len(feature_names))]

            mean = all_features.mean(axis=0)
            std = all_features.std(axis=0)

            # pooled std, which should be used for Cohen's d (cf. https://en.wikipedia.org/wiki/Effect_size#Cohen's_d)
            dementia_var, control_var = features_dementia.var(axis=0, ddof=1), features_control.var(axis=0, ddof=1)
            dementia_n, control_n = features_dementia.shape[0], features_control.shape[0]
            pooled_std = np.sqrt(((dementia_n - 1) * dementia_var + (control_n - 1) * control_var) / (dementia_n + control_n - 2))

            # normalized values
            dementia_normalized = (features_dementia - mean) / pooled_std
            control_normalized = (features_control - mean) / pooled_std

            # diff
            diff = dementia_normalized.mean() - control_normalized.mean()
            sorted_features = list(diff.sort_values(key=abs, ascending=False).index)

            # sorted
            dementia_normalized = dementia_normalized[sorted_features]
            control_normalized = control_normalized[sorted_features]
            diff = diff[sorted_features]

            fig, (ax, ax2) = plt.subplots(2, 1, figsize=(len(feature_names) / 4 + 2, 8), height_ratios=(1, 2))
            fig.suptitle(f'Distribution of normalized feature values Dementia vs. Control')
            ax2.boxplot(dementia_normalized.values, sym="", positions=positions_dementia,
                                         showmeans=True, meanline=True,
                                         patch_artist=True, boxprops=dict(facecolor="#ffa8a8"))
            ax2.boxplot(control_normalized.values, sym="", positions=positions_control,
                                        showmeans=True, meanline=True,
                                        patch_artist=True, boxprops=dict(facecolor="#a8d9ff"))
            ax2.set_xticks(tick_positions, sorted_features, rotation=45, ha='right', rotation_mode='anchor')
            ax2.set_ylabel("Normalized value of feature")
            ax2.axhline(0, linestyle=":", color="k", alpha=0.5)

            blue_patch = mpatches.Patch(color='#ffa8a8', label='Dementia')
            green_patch = mpatches.Patch(color='#a8d9ff', label='Control')
            ax2.legend(handles=[blue_patch, green_patch])

            ax.plot(tick_positions, np.abs(diff), ".-", color="k", linewidth=1)
            ax.set_xticks([], [])
            ax.set_ylabel("Effect size [Cohen's d]")
            #ax.set_ylim([0, 1.6])
            ax.set_xlim(ax2.get_xlim()[0], ax2.get_xlim()[1])

            plt.tight_layout()
            plt.savefig(os.path.join(self.run_parameters.results_dir, "feature_distribution.png"))
            plt.close()

        def plot_individual_distribution(features_dementia, features_control, all_features, figsize_individual):
            feature_names = all_features.columns

            n_cols = int(np.ceil(np.sqrt(len(feature_names))))
            n_rows = int(np.ceil(len(feature_names) / n_cols))
            print("n_rows, n_cols", n_rows, n_cols)
            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(n_cols * figsize_individual[0], n_rows * figsize_individual[1]))

            if n_rows == 1 and n_cols == 1:
                axes = [[axes]]
            elif n_rows == 1 or n_cols == 1:
                axes = [axes]

            # mean and std, normalize values, calculate difference, sort features accordingly
            mean = all_features.mean(axis=0)
            std = all_features.std(axis=0)

            dementia_normalized = (features_dementia - mean) / std
            control_normalized = (features_control - mean) / std

            diff = dementia_normalized.mean() - control_normalized.mean()
            sorted_features = list(diff.sort_values(key=abs, ascending=False).index)

            features_dementia = features_dementia[sorted_features]
            features_control = features_control[sorted_features]

            for feature, ax in zip(sorted_features, [col for row in axes for col in row]):
                ax.boxplot(features_dementia[feature], sym="", positions=[0],
                           showmeans=True, meanline=True)
                ax.boxplot(features_control[feature], sym="", positions=[1],
                           showmeans=True, meanline=True)
                ax.set_xticks([0, 1], ['Dementia', 'Control'], rotation=45, ha='right', rotation_mode='anchor')
                ax.set_ylabel("Feature Value")
                ax.set_title(feature, fontsize=10)
                ax.axhline(mean[feature], linestyle=":", color="k", alpha=0.5)

            plt.tight_layout()
            plt.savefig(os.path.join(self.run_parameters.results_dir, "feature_distribution_individual.png"))
            plt.close()

        df = dataset.data
        df['label'] = dataset.labels
        all_features = df.drop(columns=['label'])
        features_dementia = df.query("label == 1").drop(columns=['label'])
        features_control = df.query("label == 0").drop(columns=['label'])

        plot_distribution(features_dementia, features_control, all_features)
        plot_individual_distribution(features_dementia, features_control, all_features, figsize_individual=(2, 2))


class RandomForest(TreeBasedClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__("RandomForest", *args, **kwargs)

class GradientBoosting(TreeBasedClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__("GradientBoosting", *args, **kwargs)
