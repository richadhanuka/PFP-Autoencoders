
### input: protein sequence, output: rows X 400 frequency matrix
def NFV400(data_sequence):
    ## 2 mer list
    amino_acid=['C', 'D', 'S', 'Q', 'K', 'P', 'T', 'F', 'A', 'G', 'I', 'E', 'L', 'H', 'R', 'W', 'M', 'N', 'Y', 'V']
    nmer_list = list()
    for i in range(20):
        for j in range(20):
            nmer=amino_acid[i]+amino_acid[j]
            nmer_list.append(nmer)
                
    ####generates frequency matrix
    freq_mat=[]            
    for seq in range(data_sequence.shape[0]):           
        protein = data_sequence[seq]
        counts = {}
        myData=[]
        for nmer in nmer_list: 
            counts[nmer] = 0
        for nmer in nmer_list:
            if nmer in protein:
                counts[nmer] = protein.count(nmer)
                fraction = float(counts[nmer]) / float(len(protein))
                percent = fraction                    
            else:
                percent =0
            myData.append(percent)
        freq_mat.append(myData)
    return freq_mat


### input: protein sequence, output: rows X 8000 frequency matrix    
def NFV8000(data_sequence):
    ## 3 mer list
    amino_acid=['C', 'D', 'S', 'Q', 'K', 'P', 'T', 'F', 'A', 'G', 'I', 'E', 'L', 'H', 'R', 'W', 'M', 'N', 'Y', 'V']
    nmer_list = list()
    for i in range(20):
        for j in range(20):
            for k in range(20):
                nmer=amino_acid[i]+amino_acid[j]+amino_acid[k]
                nmer_list.append(nmer)
                
    ####generates frequency matrix
    freq_mat=[]            
    for seq in range(data_sequence.shape[0]):           
        protein = data_sequence[seq]
        counts = {}
        myData=[]
        for nmer in nmer_list: 
            counts[nmer] = 0
        for nmer in nmer_list:
            if nmer in protein:
                counts[nmer] = protein.count(nmer)
                fraction = float(counts[nmer]) / float(len(protein))
                percent = fraction                    
            else:
                percent =0
            myData.append(percent)
        freq_mat.append(myData)
    return freq_mat

###sample test
import numpy as np        
a = ["CDSFG"]
f = NFV400(np.array(a))