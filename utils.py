from collections import defaultdict
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt


# kgrams sparse feature matrix
def kgram_sparse_matrix(df, data_column='seg', kgram_level=6):
    feature_set = {}
    id_row = 0
    feature_id = 0
    row = []
    col = []
    data = []
    for index, row in df.iterrows():
        local_features = defaultdict(lambda: 0)
        sequence = df[data_column]
        for kgram in kgram_level + 1:
            for i in range(len(sequence) - kgram):
                sub_seq = row.seq[i:i + kgram]
                local_features[sub_seq] += 1
        for key in local_features.keys():ww
            if key not in feature_set:
                feature_set[key] = feature_id
                feature_id += 1
            row.append(id_row)
            col.append(feature_set[key])
            data.append(local_features[key])
        id_row += 1
    return csr_matrix((data, (row, col)), shape=(len(df), feature_id + 1))


# matplotlib support
def show_feature_extraction_memory_usage(x, time, memory, title):
    desc = '_kgrams_time_memory_usage.png'
    fig, axes = plt.subplots(2, sharex=True)
    axes[0].set_title('Feature extraction: ')
    axes[0].set_ylabel('Time (s)')
    axes[0].plot(x, time, 'b')
    axes[1].set_title('Memory usage:')
    axes[1].set_xlabel('Max k gram')
    axes[1].set_ylabel('Bytes')
    axes[1].plot(x, memory, 'b')
    plt.savefig('doc/imgs/' + title + desc)
    plt.show()


def show_accuracy_learning_time(names, x, accuracy, learning_time, title):
    desc = '_kgrams_accuracy_time.png'
    fig, axes = plt.subplots(2, sharex=True)
    axes[0].set_title('Accuracy:')
    axes[0].set_ylabel('%')
    axes[0].set_ylim((0, 100))
    for idx, name in enumerate(names):
        axes[0].plot(x, [100 * x for x in accuracy[name]], 'C' + str(idx), label=name)
    axes[1].set_title('Learning time:')
    axes[1].set_xlabel('Max k gram')
    axes[1].set_ylabel('Time (s)')
    for idx, name in enumerate(names):
        axes[1].plot(x, learning_time[name], 'C' + str(idx))
    lgd = fig.legend(bbox_to_anchor=(0.9, 0.7), loc=2, borderaxespad=0.)
    plt.savefig('doc/imgs/' + title + desc, additional_artists=[lgd], bbox_inches='tight')
    plt.show()


def compare_accuracy_learning_time(x_1, accuracy_1, time_1, label_1, x_2, accuracy_2, time_2, label_2, title):
    desc = '_comparison.png'
    fig, axes = plt.subplots(2, sharex=True)
    axes[0].set_title('Feature extraction: ')
    axes[0].set_ylabel('Time (s)')
    axes[0].plot(x_1, accuracy_1, 'b')
    axes[0].plot(x_2, accuracy_2, 'r')
    axes[1].set_title('Memory usage:')
    axes[1].set_xlabel('Max k gram')
    axes[1].set_ylabel('Bytes')
    axes[1].plot(x_1, time_1, 'b', label=label_1)
    axes[1].plot(x_2, time_2, 'r', label=label_2)
    lgd = fig.legend(bbox_to_anchor=(0.9, 0.7), loc=2, borderaxespad=0.)
    plt.savefig('doc/imgs/' + title + desc, additional_artists=[lgd], bbox_inches='tight')
    plt.show()
