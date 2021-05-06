import numpy as np


def parse_epoch_loss(fpath):
    train_losses = []
    test_losses = []
    with open(fpath,'r') as f:
        for line in f:
            if not (line.startswith("INFO") and "Epoch" in line and "loss" in line):
                continue

            s = line.strip()

            train_loss = s[s.find("Train loss: ") + len("Train loss: (") : s.find(")")]
            test_loss = s[s.find("Test loss ") + len("Test loss (") : -1]

            train_loss = [float(loss) for loss in train_loss.split(", ")]
            test_loss = [float(loss) for loss in test_loss.split(", ")]

            train_losses.append(train_loss)
            test_losses.append(test_loss)
    train_losses = np.array(train_losses)
    test_losses = np.array(test_losses)
    return train_losses, test_losses


def parse_batch_loss(fpath, train_epoch_len=1000, test_epoch_len=100):
    if not (train_epoch_len > 0 and test_epoch_len > 0):
        raise ValueError('train_epoch_len and test_epoch_len must be positive')
    if not (isinstance(train_epoch_len, int) and isinstance(test_epoch_len, int)):
        raise TypeError('train_epoch_len and test_epoch_len must be int')
    train_losses = []
    test_losses = []
    with open(fpath, 'r') as f:
        cur_epoch = 0
        batch_train_losses = []
        batch_test_losses = []
        for line in f:
            if not(line.startswith("INFO") and "epoch" in line and "(loss: " in line):
                continue
            
            s = line.strip()
            epoch = int(s[s.find("epoch ") + len("epoch ") : ])
            if cur_epoch == 0:
                cur_epoch = epoch

            elif cur_epoch > 0 and epoch != cur_epoch:
                assert len(batch_train_losses) >= train_epoch_len, len(batch_train_losses)
                assert len(batch_test_losses) >= test_epoch_len, len(batch_test_losses)

                cur_epoch = epoch
                train_losses.append(batch_train_losses[-train_epoch_len:])
                test_losses.append(batch_test_losses[-test_epoch_len:])
                batch_train_losses.clear()
                batch_test_losses.clear()

            if "Train" in line:
                train_loss = float(s[s.find("Train (loss: ") + len("Train (loss: ") : s.find(")")])
                batch_train_losses.append(train_loss)
            elif "Test" in line:
                test_loss = float(s[s.find("Test (loss: ") + len("Test (loss: ") : s.find(")")])
                batch_test_losses.append(test_loss)
            else:
                raise NotImplementedError('Reached an unexpected line')

    train_losses = np.array(train_losses)
    test_losses = np.array(test_losses)
    return train_losses, test_losses
