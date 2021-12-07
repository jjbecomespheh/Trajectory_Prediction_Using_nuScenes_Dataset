import matplotlib.pyplot as plt

# Plots the training and validation loss
def plot_loss(tl, vl): 
    plt.figure(figsize=(10,10))
    epovec=range(len(tl))
    plt.plot(epovec,tl,epovec,vl,linewidth=3)
    plt.legend(('Train_loss','Val_loss'))

    # make the graph understandable: 
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.show()

# Plots results of trajectory prediction
def plot_results(num, predicted_data, data_expected, data_original=None, save=False):
    original_plot_x, original_plot_y = [], []
    expected_plot_x, expected_plot_y = [], []
    data_predict_x, data_predict_y = [], []

    for i in data_expected:
        expected_plot_x.append(i[0])
        expected_plot_y.append(i[1])
    for i in predicted_data:
        data_predict_x.append(i[0])
        data_predict_y.append(i[1])
    if data_original is not None:
        for i in data_original:
            original_plot_x.append(i[0])
            original_plot_y.append(i[1])
        plt.scatter(original_plot_x,original_plot_y, marker=".",color="b",label='Input')
    

    ulti_min_x = min(min(original_plot_x),min(expected_plot_x))
    ulti_min_y = min(min(original_plot_y),min(expected_plot_y))
    ulti_max_x = max(max(original_plot_x),max(expected_plot_x))
    ulti_max_y = max(max(original_plot_y),max(expected_plot_y))

    plt.scatter(expected_plot_x,expected_plot_y, marker=".", color="g",label='Expected')
    plt.scatter(data_predict_x,data_predict_y, marker=".", color="r",label='Predicted')
    plt.scatter(expected_plot_x[0],expected_plot_y[0], color="b", marker="s", s=200, label='Car')
    plt.xlim(ulti_min_x-10, ulti_max_x+10)
    plt.ylim(ulti_min_y-10, ulti_max_y+10)
    plt.suptitle('Time-Series Prediction')
    plt.legend()

    if save:
        plt.savefig("pictures/myfig"+str(num)+".jpeg")

    plt.show()

