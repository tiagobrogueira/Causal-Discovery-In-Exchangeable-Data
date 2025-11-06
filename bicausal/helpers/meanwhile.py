#Methods return one number which is the score: If the score is negative it is the opposite direction: Y->X
def test_tuebingen(func,*args, **kwargs):
    data, weights = getTuebingen() #labels are always one!
          
    scores =np.array([func(d,*args, **kwargs) for d in data])
    weights = weights[~np.isnan(scores)]  
    scores=scores[~np.isnan(scores)]  # Remove NaN scores
    #AUROC
    

    #plot
    #dec_score = plot_decision(scores,"tuebingen",auroc, accuracy, xs, ys, absolut, relative, func.__name__,**kwargs)
    #save_excel(func.__name__,"tuebingen",[auroc,accuracy,absolut,relative,dec_score],["Auroc","Accuracy","Absolute","Relative","DecScore"])
    return auroc, accuracy, alameda