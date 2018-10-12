import sys
import os


def read_words(infile):
    answer = []
    with open(infile, "r", encoding="iso-8859-1") as fin:
        for line in fin:
            line = line.strip()
            if line == "":
                continue
            answer.append(line.split()[0])
    return sorted(answer)


def read_pairs(infile):
    answer = []
    with open(infile, "r", encoding="iso-8859-1") as fin:
        for line in fin:
            line = line.strip()
            if line == "":
                continue
            line = line.split(",")[0]
            splitted = line.split()
            word, morphemes = splitted[0], splitted[1:]
            morphemes = [x.split(":")[0] for x in morphemes]
            morphemes = [x for x in morphemes if x != "~"]
            answer.append((word, morphemes))
    return sorted(answer)


def extract_pairs_for_words(words, pairs):
    i, j = 0, 0
    pairs_with_words = []
    while i < len(words) and j < len(pairs):
        if words[i] > pairs[j][0]:
            j += 1
        else:
            if words[i] < pairs[j][0]:
                print(words[i])
            else:
                pairs_with_words.append(pairs[j])
                j += 1
            i += 1
    return pairs_with_words


if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit("Usage: python <dir_with_morphochallenge_files> <languages> <outdir>")
    input_dir, languages, output_dir = sys.argv[1:]
    languages = languages.split(",")
    for language in languages:
        infile_all = os.path.join(input_dir, "goldstd_combined.segmentation.{}".format(language))
        infile_train_labels = os.path.join(input_dir, "goldstd_trainset.labels.{}".format(language))
        train_words = read_words(infile_train_labels)
        infile_dev_labels = os.path.join(input_dir, "goldstd_develset.labels.{}".format(language))
        dev_words = read_words(infile_dev_labels)
        all_pairs = read_pairs(infile_all)
        train_pairs = extract_pairs_for_words(train_words, all_pairs)
        dev_pairs = extract_pairs_for_words(dev_words, all_pairs)
        outfile = os.path.join(output_dir, "{}.train".format(language))
        with open(outfile, "w", encoding="utf8") as fout:
            for word, morphemes in train_pairs:
                fout.write("{}\t{}\n".format(word, "/".join(morphemes)))
        outfile = os.path.join(output_dir, "{}.dev".format(language))
        with open(outfile, "w", encoding="utf8") as fout:
            for word, morphemes in dev_pairs:
                fout.write("{}\t{}\n".format(word, "/".join(morphemes)))