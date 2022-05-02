from sklearn.metrics import accuracy_score, f1_score

def flat_accuracy(labels, preds):    
    labels = labels.cpu().deteach().numpy()
    preds = preds.cpu().deteach().numpy()
    return {
        'accuracy': accuracy_score(labels, preds), 
        'macro_f1_score': f1_score(labels, preds, average='macro'), 
        'micro_f1_score': f1_score(labels, preds, average='micro')
    }