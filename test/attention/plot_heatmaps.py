import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def plot(a, b, titles, out_file_name=None):
    assert isinstance(titles, list)
    assert len(titles) == 3

    max_score = np.max([np.max(a), np.max(b)])
    c = a - b

    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    cbar_ax = fig.add_axes([.91, 0.11, .03, .77])
    cbar_ax.tick_params(labelsize=16)
    axes[0].set_title(titles[0], fontdict={'fontsize': 16})
    h = sns.heatmap(a.reshape(-1, 1), annot=True, ax=axes[0], vmax=max_score, vmin=0, yticklabels=range(1, len(a) + 1),
                xticklabels=False, cbar=True, cbar_ax=cbar_ax, annot_kws={"size": 14})
    h.set_yticklabels(h.get_yticklabels(), fontsize=16, rotation=0)
    axes[0].set_ylabel('layers', fontsize=14)
    axes[1].set_title(titles[1], fontdict={'fontsize': 16})
    sns.heatmap(b.reshape(-1, 1), annot=True, ax=axes[1], vmax=max_score, vmin=0, xticklabels=False, yticklabels=False,
                cbar=False, cbar_ax=None, annot_kws={"size": 14})
    axes[2].set_title(titles[2], fontdict={'fontsize': 16})
    sns.heatmap(c.reshape(-1, 1), annot=True, ax=axes[2], vmax=max_score, vmin=0, xticklabels=False, yticklabels=False,
                cbar=False, cbar_ax=None, annot_kws={"size": 14})
    plt.subplots_adjust(wspace=0.2)

    if out_file_name is not None:
        plt.savefig(out_file_name, bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    # match_sent_pretrain = np.array([0.05, 0.06, 0.11, 0.16, 0.13, 0.07, 0.06, 0.01, 0.08, 0.34, 0.38, 0.21])
    # non_match_sent_pretrain = np.array([0.03, 0.05, 0.08, 0.12, 0.13, 0.05, 0.05, 0, 0.06, 0.23, 0.27, 0.11])
    # plot(match_sent_pretrain, non_match_sent_pretrain, ['Match', 'Non match', 'Diff'])

    # all_sent_pretrain = np.array([0.04, 0.06, 0.10, 0.14, 0.13, 0.06, 0.06, 0.0, 0.07, 0.29, 0.33, 0.16])
    # all_attr_pretrain = np.array([0.04, 0.06, 0.11, 0.16, 0.15, 0.09, 0.10, 0.04, 0.1, 0.29, 0.37, 0.24])
    # plot(all_attr_pretrain, all_sent_pretrain, ['Attr-pair', 'Sent-pair', 'Diff'])

    # match_attr_pretrain = np.array([0.05, 0.07, 0.12, 0.18, 0.15, 0.1, 0.11, 0.06, 0.11, 0.32, 0.41, 0.27])
    # non_match_attr_pretrain = np.array([0.06, 0.14, 0.16, 0.18, 0.19, 0.17, 0.2, 0.04, 0.17, 0.29, 0.24, 0.18])
    # plot(match_attr_pretrain, non_match_attr_pretrain, ['Match', 'Non match', 'Diff'])

    # match_sent_fine = np.array([0.05, 0.07, 0.12, 0.12, 0.17, 0.07, 0.04, 0.04, 0.07, 0.18, 0.18, 0.09])
    # non_match_sent_fine = np.array([0.03, 0.06, 0.09, 0.05, 0.05, 0.02, 0.02, 0.03, 0.04, 0.07, 0.06, 0.04])
    # plot(match_sent_fine, non_match_sent_fine, ['Match', 'Non match', 'Diff'])

    # non_match_sent_fine = np.array([0.03, 0.06, 0.09, 0.05, 0.05, 0.02, 0.02, 0.03, 0.04, 0.07, 0.06, 0.04])
    # fp_sent_fine = np.array([0.03, 0.09, 0.11, 0.08, 0.12, 0.06, 0.07, 0.08, 0.08, 0.07, 0.08, 0.05])
    # plot(non_match_sent_fine, fp_sent_fine, ['Non match', 'FP', 'Diff'])
    #
    # match_sent_fine = np.array([0.05, 0.07, 0.12, 0.12, 0.17, 0.07, 0.04, 0.04, 0.07, 0.18, 0.18, 0.09])
    # fn_sent_fine = np.array([0.07, 0.09, 0.12, 0.07, 0.1, 0.06, 0.04, 0.04, 0.05, 0.05, 0.08, 0.07])
    # plot(match_sent_fine, fn_sent_fine, ['Match', 'FN', 'Diff'])

    # all_sent_pretrain = np.array([0.04, 0.06, 0.10, 0.14, 0.13, 0.06, 0.06, 0.0, 0.07, 0.29, 0.33, 0.16])
    # all_sent_fine = np.array([0.04, 0.06, 0.11, 0.09, 0.12, 0.04, 0.03, 0.04, 0.05, 0.13, 0.12, 0.07])
    # plot(all_sent_pretrain, all_sent_fine, ['Pre-training', 'Fine-tuning', 'Diff'],
    #      out_file_name="pretrain_vs_finetune.pdf")
    # plot(all_sent_fine, all_sent_pretrain, ['Fine-tuning', 'Pre-training', 'Diff'])

    ucs = ["Structured_Fodors-Zagats", "Structured_DBLP-GoogleScholar", "Structured_DBLP-ACM",
     "Structured_Amazon-Google", "Structured_Walmart-Amazon", "Structured_Beer",
     "Structured_iTunes-Amazon", "Textual_Abt-Buy", "Dirty_iTunes-Amazon", "Dirty_DBLP-ACM",
     "Dirty_DBLP-GoogleScholar", "Dirty_Walmart-Amazon"]

    import matplotlib.image as mpimg

    imgs = [mpimg.imread(f'{uc}.png') for uc in ucs]
    nrows = 3
    ncols = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 12))
    for i in range(len(imgs)):
        ax = axes[i // ncols][i % ncols]
        ax.imshow(imgs[i])
        ax.set_axis_off()
        ax.autoscale(False)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    # plt.show()
    plt.savefig("full.pdf", bbox_inches='tight')
