import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def date_time(x):
    """Get formatted datetime string"""
    if x == 1:
        return 'Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    if x == 2:    
        return 'Timestamp: {:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now())
    if x == 3:  
        return 'Date now: %s' % datetime.datetime.now()
    if x == 4:  
        return 'Date today: %s' % datetime.date.today()

def plot_performance(history, figure_directory=None, ylim_pad=[0, 0]):
    """Plot training and validation accuracy/loss"""
    xlabel = 'Epoch'
    legends = ['Training', 'Validation']

    plt.figure(figsize=(20, 5))

    # Accuracy plot
    plt.subplot(121)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy\n' + date_time(1), fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.legend(legends, loc='upper left')
    plt.grid()

    # Loss plot
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss\n' + date_time(1), fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.legend(legends, loc='upper left')
    plt.grid()
    
    if figure_directory:
        plt.savefig(figure_directory + "/training_history.png")
    
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, figsize=(15, 15)):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
