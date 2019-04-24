import urllib.request
import os
import tarfile
import itertools
from collections import Counter
import torch
import numpy as np


class MnistStrokeSequence:
    def __init__(self, mode="test", shuffle=True, batch_size=1, root_dir="data", match_dimension="mean"):
        """
        :param train: true if data is to be used for training
        :param shuffle: randomly shuffle the data or not
        :param batch_size:
        :param root_dir: the directory to the save the downloaded data or where the data is already saved
        :param match_dimension: "truncate" or "pad" strategy to match dimensions
        """
        self.mode = mode
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.match_dimension = match_dimension

        self.url_sequence = "https://github.com/edwin-de-jong/mnist-digits-stroke-sequence-data/raw/master/sequences.tar.gz"
        self.sequence_fname = "sequences.tar.gz"
        self.sequence_fname = os.path.join(self.root_dir, self.sequence_fname)

        self.indices = []
        self.maybe_download()
        self.process()

    def process(self):
        """ Prepare the dataset for training, testing and validation """
        self.processed_dir = "data/processed_mnist"
        self.processed_inputs = os.path.join(self.processed_dir, "inputs.npy")
        self.processed_targets = os.path.join(self.processed_dir, "targets.npy")

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        # If the processed files exist, read them
        if os.path.exists(self.processed_inputs) and os.path.exists(self.processed_targets):
            self.targets = np.load(self.processed_targets)
            self.inputs = np.load(self.processed_inputs)

        else:
            # If you have not already processed the sequence files, you can process them now
            self.sequence_dir = "/home/wasim/Documents/sequences"
            if not os.path.exists(self.sequence_dir):
                raise ValueError("The file of sequences does not occur at ", self.sequence_dir)

            files = os.listdir(self.sequence_dir)
            files = [file for file in files if file.__contains__("targetdata")]

            self.inputs, self.targets = [], []
            for fname in files:
                fname = os.path.join(self.sequence_dir, fname)

                sequence = open(fname, mode='r', encoding='utf8').read().splitlines()
                sequence = " ".join(sequence).split()
                sequence = np.asarray(sequence, np.int32).reshape(-1, 14)

                label = sequence[:, :10]
                label = np.mean(label, axis=0)
                assert sum(label) == 1.0
                label = np.argsort(label.tolist())[-1]

                input = sequence[:, 10:].flatten()
                self.targets.append(label)
                self.inputs.append(input)

            del files, sequence, input, label

            np.save(self.processed_targets, self.targets)
            np.save(self.processed_inputs, self.inputs)

        # There are only 70000
        if self.mode == "train":
            start = 0
            end = 50000
        elif self.mode == "test":
            start = 50000
            end = 60000
        else:
            start = 60000
            end = 70000

        self.targets = self.targets[start:end]

        self.num_batches = len(self.targets) // self.batch_size
        self.indices = list(range(len(self.targets)))
        self.indices = self.indices[:self.num_batches * self.batch_size]

        self.inputs = self.inputs[start:start + len(self.indices)]

        print("Total number of samples : {} Classes : {}".format(len(self.indices), Counter(self.targets)))

    def __next__(self):
        """ Return the next batch of data for training """
        batch_indices = [self.indices.pop(0) for _ in range(self.batch_size)]
        inputs = [self.inputs[index] for index in batch_indices]
        targets = [self.targets[index] for index in batch_indices]

        input_lens = []
        if self.batch_size > 1:
            # Sorting everything according to decrease length of sequence
            input_lens = [len(seq) for seq in inputs]

            if self.match_dimension == "pad":
                input_lens = sorted(input_lens, reverse=True)
                ranks = np.argsort(input_lens)
                inputs = [inputs[rank] for rank in ranks]
                targets = [targets[rank] for rank in ranks]
                inputs = self.pad(inputs)

            elif self.match_dimension == "truncate":
                inputs = [input[:min(input_lens)] for input in inputs]

            elif self.match_dimension == "mean":
                mean_len = int(np.mean(input_lens) // 4) * 4
                inputs = [input[:mean_len] for input in inputs]
                inputs = self.pad(inputs)

            else:
                raise ValueError("Choose among : mean, pad and truncate for dimension matching strategy")

        # Convert the array into tensors
        inputs = torch.tensor(inputs, dtype=torch.float).reshape(self.batch_size, -1, 4)
        targets = torch.tensor(targets)

        if len(self.indices) < self.batch_size:
            self.indices = list(range(len(self.targets)))

        return inputs, input_lens, targets

    def next(self):
        return self.__next__()

    def maybe_download(self):
        """ Download the files if they do not exist """
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

        if not os.path.exists(self.sequence_fname):
            print("Downloading ... ", self.sequence_fname)
            urllib.request.urlretrieve(self.url_sequence, self.sequence_fname)
            # print("Extracting tar.gz and saving them at the same location")
            # tarfile.open(self.sequence_fname).extractall(self.root_dir)

    def pad(self, l, fillvalue=0):
        return list(itertools.zip_longest(*l, fillvalue=fillvalue))

    def __len__(self):
        return self.num_batches


if __name__ == '__main__':
    testLoader = MnistStrokeSequence(mode="test", shuffle=True, batch_size=64, match_dimension="mean")
    for _ in range(2):
        for i in range(len(testLoader)):
            inputs, lens, targets = testLoader.next()
            print(i, len(testLoader), np.mean(lens))
    print(inputs.size(), lens, max(lens))
