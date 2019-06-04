import sys
import getopt

from read import read_splitted


def measure_quality(targets, predicted_targets, english_metrics=False, measure_last=True):
    """

    targets: метки корректных ответов
    predicted_targets: метки предсказанных ответов

    Возвращает словарь со значениями основных метрик
    """
    measure_last = False
    TP, FP, FN, equal, total = 0, 0, 0, 0, 0
    SE = ['{}-{}'.format(x, y) for x in "SE" for y in ["ROOT", "PREF", "SUFF", "END", "LINK", "None"]]
    # SE = ['S-ROOT', 'S-PREF', 'S-SUFF', 'S-END', 'S-LINK', 'E-ROOT', 'E-PREF', 'E-SUFF', 'E-END']
    corr_words = 0
    for i, (corr, pred) in enumerate(zip(targets, predicted_targets)):
        corr_len = len(corr) + int(measure_last) - 1
        pred_len = len(pred) + int(measure_last) - 1
        boundaries = [i for i in range(corr_len) if corr[i] in SE]
        pred_boundaries = [i for i in range(pred_len) if pred[i] in SE]
        common = [x for x in boundaries if x in pred_boundaries]
        TP += len(common)
        FN += len(boundaries) - len(common)
        FP += len(pred_boundaries) - len(common)
        # if len(pred_boundaries) == 0 and len(boundaries) == 0:
        #     TP += 1
        equal += sum(int(x==y) for x, y in zip(corr, pred))
        total += len(corr)
        corr_words += (corr == pred)
    metrics = ["Точность", "Полнота", "F1-мера", "Корректность", "Точность по словам"]
    if english_metrics:
        metrics = ["Precision", "Recall", "F1", "Accuracy", "Word accuracy"]
    results = [TP / (TP+FP), TP / (TP+FN), TP / (TP + 0.5*(FP+FN)),
               equal / total, corr_words / len(targets)]
    answer = list(zip(metrics, results))
    return answer

if __name__ == "__main__":
    use_morphs, morph_sep = False, "/"
    opts, (gold_file, pred_file) = getopt.getopt(sys.argv[1:], "ms:")
    for opt, val in opts:
        if opt == "-m":
            use_morphs = True
        elif opt == "-s":
            morph_sep = val
    if not use_morphs:
        _, gold_data = read_splitted(gold_file, transform_to_BMES=True, morph_sep=morph_sep, shuffle=False)
        _, pred_data = read_splitted(pred_file, transform_to_BMES=True, morph_sep=morph_sep, shuffle=False)
        quality = measure_quality(gold_data, pred_data, measure_last=False)
        for key, value in sorted(quality):
            print("{}={:.2f}".format(key, 100 * value))


