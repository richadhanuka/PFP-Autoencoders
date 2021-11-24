from sklearn.metrics import hamming_loss
#from sklearn.metrics import classification_report

def metrics(Y_predicted, Y_test):
    totalPrecision = 0
    totalRecall = 0
    totalF1Score = 0
    for i in range(Y_test.shape[0]):
        truePositive = 0
        trueNegative = 0
        falsePositive = 0
        falseNegative = 0
        precision = 0
        recall = 0
        for j in range(Y_test.shape[1]):
            if Y_predicted[i,j] == 1:
                if Y_test[i,j] == 1:
                    truePositive = truePositive + 1
                else:    
                    falsePositive = falsePositive + 1
            else:
                if Y_test[i,j] == 1:
                    falseNegative = falseNegative + 1
                else:
                    trueNegative = trueNegative + 1
        try:
            precision = truePositive/(truePositive + falsePositive)
        except ZeroDivisionError:
            precision = 0               
        
        totalPrecision = totalPrecision + precision
        try:
            recall = truePositive/(truePositive + falseNegative)
        except ZeroDivisionError:
            recall = 0
        
        totalRecall = totalRecall + recall
        try:
            f1Score = 2 * precision * recall / (precision + recall)
        except:
            f1Score = 0
        totalF1Score = totalF1Score + f1Score
        
    avgPrecision = totalPrecision / (Y_test.shape[0])
    avgRecall = totalRecall / (Y_test.shape[0])
    avgF1Score = totalF1Score / (Y_test.shape[0])
    #print("optimizer=Adam,epochs=300,batch_size=32")
    #print("Average Precision : " + str(avgPrecision))
    #print(avgPrecision)
    #print("Average Recall : "  + str (avgRecall))
    #print(avgRecall)
    #print("Average F1-Score : " + str(avgF1Score))
    #print(avgF1Score)
    try:
        F1Score = ( 2 * avgPrecision * avgRecall ) / ( avgPrecision + avgRecall )
    except:
        F1Score = 0#print("F1-score : " + str(F1Score))
    hammingLoss = hamming_loss(Y_predicted, Y_test)
    #print("Hamming Loss : " + str(hammingLoss))
    return avgPrecision, avgRecall, avgF1Score, F1Score, hammingLoss
    #print(classification_report(Y_test ,Y_predicted ))