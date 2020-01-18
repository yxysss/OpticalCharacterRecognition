from utils.plotter import Plotter
import csv


if __name__ == "__main__":
    history1 = "experiments/2019-12-25/xy_emnist_from_config/history_params/parameters.csv"
    history2 = "experiments/2019-12-25/xy_emnist_x_from_config/history_params/parameters.csv"
    history3 = "experiments/2019-12-26/xy_emnist_bn_from_config/history_params/parameters.csv"
    history_list = [history1, history2, history3]
    index = 0
    for history in history_list:
        index += 1
        file = open(history, "r")
        reader = csv.reader(file)
        epochs = []
        acc = []
        val_acc = []
        loss = []
        val_loss = []
        for item in reader:
            if reader.line_num == 1:
                continue
            epochs.append(int(item[0]))
            acc.append(float(item[1]))
            loss.append(float(item[2]))
            val_acc.append(float(item[3]))
            val_loss.append(float(item[4]))
        Plotter.plotgraph(title="model_"+str(index), epochs=epochs, acc=acc, val_acc=val_acc)
        Plotter.plotlossgraph(title="model_"+str(index), epochs=epochs, loss=loss, val_loss=val_loss)

