import numpy as np
import pandas as pd
import os
import torch
from sklearn import metrics as sk_metrics
import matplotlib.pyplot as plt

from dataloader.metadata_loader import ADReSSParticipantMetadataLoader


class BiasAnalysis:
    def __init__(self, local=None, constants=None, experiment_name=""):
        self.experiment_name = experiment_name
        participant_metadata_loader = ADReSSParticipantMetadataLoader(local=local, constants=constants)
        self.participant_metadata = participant_metadata_loader.load_metadata()
        pass

    def _get_metrics_flat(self, metric_name, predictions_flat, labels_flat):
        predictions_rounded = np.round(np.array(predictions_flat))
        metrics = {
            'accuracy': sk_metrics.accuracy_score(labels_flat, predictions_rounded),
        }
        if len(np.unique(labels_flat)) > 1:
            metrics = {**metrics,
                       'recall / sensitivity': sk_metrics.recall_score(labels_flat, predictions_rounded),
                       'specificity': sk_metrics.recall_score(labels_flat, predictions_rounded, pos_label=0),
                       'precision': sk_metrics.precision_score(labels_flat, predictions_rounded),
                       'auroc': sk_metrics.roc_auc_score(labels_flat, predictions_flat),
                       'average_precision': sk_metrics.average_precision_score(labels_flat, predictions_flat),
                       'log_loss': sk_metrics.log_loss(labels_flat, predictions_flat)}

        try:
            res = metrics[metric_name]
        except:
            print(f"Cannot calculate metric {metric_name} on bootstrap sample. Sample might contain only one label. Continue...")
            res = None
        return res

    def _get_boostrapped_metrics(self, group_df, metric_name, n_bootstrap_samples):
        # draw bootstrap samples, calculate metric for each sample, return metrics
        if metric_name != 'accuracy' and len(np.unique(group_df.label)) == 1:
            raise ValueError(f"Cannot calculate {metric_name} if there is only one label in the data")
        metrics = []
        for i in range(n_bootstrap_samples):
            sample = group_df.sample(n=group_df.shape[0], replace=True, axis=0)
            if metric_name == 'wer':
                sample_metric = sample.WER.mean()
            else:
                sample_metric = self._get_metrics_flat(metric_name, sample.prediction, sample.label)
            if sample_metric is not None:
                metrics.append(sample_metric)
        return metrics

    def _flatten_data(self, *data):
        # flatten list of splits
        assert np.all([len(data_item) == len(data[0]) for data_item in data]), "Shape mismatch of outer list"
        for i in range(1, len(data)):
            for j in range(len(data[0])):
                assert np.all([len(data[i][j]) == len(data[0][j])]), "Shape mismatch of inner list"

        data_flattened = []
        for data_item in data:
            flat = [p for split in data_item for p in split]
            # move tensors to cpu if given
            flat = np.array([p.cpu() if type(p) == torch.Tensor else p for p in flat])
            data_flattened.append(flat)

        return data_flattened

    def _show_violin_plots_criteria(self, metric_name, predictions, labels, sample_names_with_metadata, criteria,
                                    n_bootstrap_samples, plot_path):
        df = sample_names_with_metadata.copy()

        df['prediction'] = predictions
        df['label'] = labels

        distributions = []
        labels = []
        group_sizes = []
        criterium_sizes = []
        ad_control_fracs = []
        metric_name_changes = []


        for crit_idx, criterium in enumerate(criteria):
            metric_name_here = metric_name  # metric can be changed for certain criteria
            if criterium == 'gender':
                groups = {'Female': df.query("gender == 1"), 'Male': df.query("gender == 0")}
            elif criterium == 'age':
                # Groups according to ADReSS dataset statistics -> end up in roughly the same numbers
                groups = {
                    'Age<62': df.query("age < 62"),
                    'Age=62-66': df.query("age >= 62 and age < 67"),
                    'Age=67-71': df.query("age >= 67 and age < 72"),
                    'Age>=72': df.query("age >= 72")
                }
            elif criterium == 'mmse':
                # Groups according to ADReSS dataset statistics -> end up in roughly the same numbers
                df_subset = df[~df.mmse.isna()].copy()
                assert df_subset.mmse.dtype == 'int'
                groups = {
                    'MMSE<19':df_subset.query("mmse < 19"),
                    'MMSE=19-28': df_subset.query("mmse < 29 and  mmse >= 19"),
                    'MMSE>=29': df_subset.query("mmse >= 29")
                }
                metric_name_here = 'accuracy'  # AUROC doesnt work for MMSE, since MMSE 29/30 only has controls
                metric_name_changes.append({
                    'start_position': np.sum(criterium_sizes) + crit_idx - 1,
                    'metric_name': metric_name_here
                })
            else:
                raise ValueError()
            criterium_sizes.append(len(groups))

            for group_name in groups:
                distributions.append(self._get_boostrapped_metrics(groups[group_name], metric_name_here,
                                                                   n_bootstrap_samples))
                labels.append(group_name)
                group_sizes.append(groups[group_name].shape[0] / df.shape[0] * 100)  # in percent
                n_samples = groups[group_name].shape[0]
                ad_control = (groups[group_name].query("Label == 1").shape[0] / n_samples, groups[group_name].query("Label == 0").shape[0] / n_samples)
                ad_control_fracs.append(ad_control)

        # plot positions s.t. there are gaps between the criteria
        criterium_sizes_cumulative = np.cumsum(criterium_sizes)
        positions = [pos for start, end, offset in zip([0, *criterium_sizes_cumulative[:-1]], criterium_sizes_cumulative, range(len(criterium_sizes_cumulative))) for pos in range(start+offset, end+offset)]

        fig, (ax, ax2, ax3) = plt.subplots(3,1, figsize=(7, 6), height_ratios=(3, 1, 1))

        # violin plot of performance per group
        ax.set_title(f"Bias {self.experiment_name} (n={df.shape[0]})")
        ax.violinplot(distributions, positions=positions, showextrema=False, showmedians=True)
        ax.set_xticks(positions, [])
        ax.set_ylabel(f"{metric_name}\n(bootstrap samples)", fontsize=8)
        for metric_name_change in metric_name_changes:
            # when metric changes, make this explicit in plot by adding a new label
            ax.text(metric_name_change['start_position'], np.mean(ax.get_ylim()), metric_name_change['metric_name'],
                    rotation=90, va='center', fontsize=8)

        # bar plots with group sizes
        ax2.bar(positions, group_sizes, alpha=0.5)
        [ax2.text(pos, val/2, f"{int(val)}%", ha="center", va='center', fontsize=8) for pos, val in zip(positions, group_sizes) if val > 0.1]
        ax2.set_ylabel("Subjects (%)", fontsize=8)
        ax2.set_xticks(positions, [])

        # bar plot with distribution of labels
        ax3.bar(positions, np.array(ad_control_fracs)[:,0], label="AD", alpha=0.5)
        [ax3.text(pos, val/2, f"{int(val*100)}%", ha="center", va='center', fontsize=8) for pos, val in zip(positions, np.array(ad_control_fracs)[:,0]) if val > 0.1]
        ax3.bar(positions, np.array(ad_control_fracs)[:,1], bottom=np.array(ad_control_fracs)[:,0], label="Ctrl", alpha=0.5)
        [ax3.text(pos, bottom + val/2, f"{int(val*100)}%", ha="center", va='center', fontsize=8) for pos, bottom, val in zip(positions, np.array(ad_control_fracs)[:,0], np.array(ad_control_fracs)[:,1]) if val > 0.1]
        ax3.set_ylim([0, 1.6])
        ax3.legend(ncol=2, fontsize=8, loc="upper right")
        ax3.set_yticks([0.5, 1])
        ax3.set_xticks(positions,labels, rotation=45, ha='right')
        ax3.set_ylabel("# AD / # Ctrl", fontsize=8)

        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close(fig)

        return




    def bias_analysis(self, predictions, labels, sample_names, plot_path=None):
        assert len(predictions) == len(labels) == len(sample_names), \
            f"Shapes dont match: predictions: {len(predictions)}, labels: {len(labels)}, sample_names {len(sample_names)}"

        # flatten data of splits into one big list of predictions / labels / samples names
        predictions_flat, labels_flat, sample_names_flat = self._flatten_data(predictions, labels, sample_names)

        sample_names_with_metadata = pd.DataFrame({'sample_name': sample_names_flat}).merge(self.participant_metadata,
                                                                                          on="sample_name", how="left")
        assert np.all([np.array(sample_names_with_metadata.sample_name) == np.array(sample_names_flat)]), \
            "Sample name order changed?"

        if plot_path is None:
            print("No path provided for bias analysis, skipping it... ")
            return

        n_bootstrap_samples = 500
        criteria = ["gender", 'age', 'mmse']
        return self._show_violin_plots_criteria('auroc', predictions_flat, labels_flat,
                                                sample_names_with_metadata, criteria,
                                                n_bootstrap_samples=n_bootstrap_samples, plot_path=plot_path)




