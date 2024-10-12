from Bio import SeqIO
from sgt import SGT
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np
import tensorflow as tf
import sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 


import argparse

# Create a parser
parser = argparse.ArgumentParser(description="IPEV")
parser.add_argument('fasta_file', type=str, help='Path to the input FASTA file.')
parser.add_argument('-filter', type=str, help='Filter option.', default='no')

# Parse command line arguments
args = parser.parse_args()

# Check the filter argument
if args.filter == 'yes':
    print("You have entered the non-virus removal functionality.")
    from Bio import SeqIO
    from sgt import SGT
    import pandas as pd
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
    import numpy as np
    import tensorflow as tf
    import sys
    import time
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt 


    start_time = time.time()
    t = time.localtime()
    current_time = time.strftime("%H_%M_%S_", t)
    try:
        # Extract the file name from the provided path
        file_local_fasta = os.path.basename(sys.argv[1])
        
        # Generate a new directory name using the current time and file name
        dir_name = 'Decontamination_'+str(current_time) + file_local_fasta
        
        # Make the directory
        os.makedirs(dir_name)

    except FileExistsError:
        print(f"Directory {dir_name} already exists!")
    except Exception as e:
        print(f"An error occurred: {e}")
        
        
    def getKmers(sequence, size):
        return [sequence[x:x+size] for x in range(len(sequence) - size + 1)]

    def sgtfunc(filename,merlen=3):
        listapl=["AAA", "AAC", "AAG", "AAT",
    "ACA", "ACC", "ACG", "ACT",
    "AGA", "AGC", "AGG", "AGT",
    "ATA", "ATC", "ATG", "ATT",
    "CAA", "CAC", "CAG", "CAT",
    "CCA", "CCC", "CCG", "CCT",
    "CGA", "CGC", "CGG", "CGT",
    "CTA", "CTC", "CTG", "CTT",
    "GAA", "GAC", "GAG", "GAT",
    "GCA", "GCC", "GCG", "GCT",
    "GGA", "GGC", "GGG", "GGT",
    "GTA", "GTC", "GTG", "GTT",
    "TAA", "TAC", "TAG", "TAT",
    "TCA", "TCC", "TCG", "TCT",
    "TGA", "TGC", "TGG", "TGT",
    "TTA", "TTC", "TTG", "TTT"]
        dictuncertain = {
            'N':'G',
            'X':'A',
            'H':'T',
            'M':'C',
            'K':'G',
            'D':'A',
            'R':'G',
            'Y':'T',
            'S':'C',
            'W':'A',
            'B':'C',
            'V':'G',
        }

        num = 0


        with open(filename,'r') as handle:


            key_store = dict()
            count_1=0
            count_2=0
            count_3=0
            count_4=0
            
            for seq_id, rec in enumerate(SeqIO.parse(handle, 'fasta')):
                address1=str(rec.seq)



                for word1, word2 in dictuncertain.items():



                    address1 = address1.replace(word1, word2)


                sequence_id = np.zeros(5)
                sequence_id[0]=seq_id
                _length_sequence = len(address1)

                if 2>len(address1[_length_sequence//1800*1800:]):
                    sequence_id[4]=_length_sequence//1800
                else:
                    sequence_id[4]=_length_sequence//1800+1
                for _num_i in range(_length_sequence//1800+1):
                    
                    sequence_id[1]=_num_i
                    if len(address1[_num_i*1800 : _num_i*1800+1800])<2:
                        pass

                    elif 2<=len(address1[_num_i*1800 : _num_i*1800+1800])<400:
                        count_1+=1
                        sequence_id[2]=len(address1[_num_i*1800 : _num_i*1800+1800])
                        sequence_id[3]=1

                        key_store[sequence_id.tobytes()]=[count_1, getKmers(address1[_num_i*1800 : _num_i*1800+1800],merlen)]
                    elif 400<=len(address1[_num_i*1800 : _num_i*1800+1800])<800:
                        count_2+=1
                        sequence_id[2]=len(address1[_num_i*1800 : _num_i*1800+1800])
                        sequence_id[3]=2
                        key_store[sequence_id.tobytes()]=[count_2, getKmers(address1[_num_i*1800 : _num_i*1800+1800],merlen)]
                    elif 800<=len(address1[_num_i*1800 : _num_i*1800+1800])<1200:
                        count_3+=1
                        sequence_id[2]=len(address1[_num_i*1800 : _num_i*1800+1800])
                        sequence_id[3]=3
                        key_store[sequence_id.tobytes()]=[count_3, getKmers(address1[_num_i*1800 : _num_i*1800+1800],merlen)]
                    elif 1200<=len(address1[_num_i*1800 : _num_i*1800+1800])<=1800:
                        count_4+=1
                        sequence_id[2]=len(address1[_num_i*1800 : _num_i*1800+1800])
                        sequence_id[3]=4
                        key_store[sequence_id.tobytes()]=[count_4, getKmers(address1[_num_i*1800 : _num_i*1800+1800],merlen)]
            
            global key_store_1 
            global key_store_2 
            global key_store_3 
            global key_store_4 
            global data_set_1
            global data_set_2
            global data_set_3
            global data_set_4
            key_store_1 = dict()
            key_store_2 = dict()
            key_store_3 = dict()
            key_store_4 = dict()
            for key_id in key_store.keys():

                if np.frombuffer(key_id)[3]==1:
                    key_store_1[key_id]=key_store[key_id]
                if np.frombuffer(key_id)[3]==2:
                    key_store_2[key_id]=key_store[key_id]
                if np.frombuffer(key_id)[3]==3:
                    key_store_3[key_id]=key_store[key_id]
                if np.frombuffer(key_id)[3]==4:
                    key_store_4[key_id]=key_store[key_id]


        sgt = SGT(alphabets=listapl,kappa=1, 
                  flatten=True, 
                  lengthsensitive=False, 
                  mode='multiprocessing')            



        try:
            corpus = pd.DataFrame(key_store_1.values(),columns=['id', 'sequence'])
            sgtpd = sgt.fit_transform(corpus)
            sgtpd.pop('id')
            data_set_1 = tf.data.Dataset.from_tensor_slices((sgtpd.values))
        except:
            pass
        try:
            corpus = pd.DataFrame(key_store_2.values(),columns=['id', 'sequence'])
            sgtpd = sgt.fit_transform(corpus)

            sgtpd.pop('id')
            data_set_2 = tf.data.Dataset.from_tensor_slices((sgtpd.values))
        except:
            pass
        try:
            corpus = pd.DataFrame(key_store_3.values(),columns=['id', 'sequence'])
            sgtpd = sgt.fit_transform(corpus)

            sgtpd.pop('id')
            data_set_3 = tf.data.Dataset.from_tensor_slices((sgtpd.values))
        except:
            pass
        try:
            corpus = pd.DataFrame(key_store_4.values(),columns=['id', 'sequence'])
            sgtpd = sgt.fit_transform(corpus)

            sgtpd.pop('id')
            data_set_4 = tf.data.Dataset.from_tensor_slices((sgtpd.values))
        except:
            pass
        return (count_1, count_2, count_3, count_4)


    ##########################################################################

    ##########################################################################

    def input_sample(input_dataset, repeat_count=1, batch_size=1, cpus=20):
        """Parses tfrecord and returns dataset for training/eval.

        Args:
            input_tfrec: Filenames of tfrecord.
            repeat_count: Number of epochs.
            batch_size: Number of examples returned per iteration.
            cpus: Number of cores used to running input pipeline.
            max_len: Length of the longest sequence.

        Returns:
            Dataset of (reads, label) pairs.
        """
        def _parse_4function(serialized):
            read = tf.reshape(serialized,[64, 64,1])
            return read
        dataset = input_dataset

        dataset = dataset.repeat(repeat_count)
        dataset = dataset.map(map_func=_parse_4function, num_parallel_calls=cpus)
        dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes=(
                                           tf.TensorShape([None,None,None])))

        return dataset
    ####################################################################

    (count_1, count_2, count_3, count_4)=sgtfunc(str(sys.argv[1]))

    try:
        model = tf.keras.models.load_model('./core/decontamination/1.hdf5')
        input_sequence1 =data_set_1
        prelabel1 = model.predict(input_sample(input_sequence1), steps=count_1)
    except:
        pass

    try:
        model = tf.keras.models.load_model('./core/decontamination/2.hdf5')

        input_sequence2 =data_set_2
        prelabel2 = model.predict(input_sample(input_sequence2), steps=count_2)

    except:
        pass
    try:
        model = tf.keras.models.load_model('./core/decontamination/3.hdf5')

        input_sequence3 =data_set_3
        prelabel3 = model.predict(input_sample(input_sequence3), steps=count_3)
    except:
        pass
    try:

        model = tf.keras.models.load_model('./core/decontamination/4.hdf5')

        input_sequence4 =data_set_4
        prelabel4 = model.predict(input_sample(input_sequence4), steps=count_4)

    except:
        pass


    try:
        pre_store_1 = dict()
        for num_i, key_id in enumerate(key_store_1.keys()):
            pre_store_1[key_id] = prelabel1[num_i]
    except:
        pass
    try:
        pre_store_2 = dict()
        for num_i, key_id in enumerate(key_store_2.keys()):
            pre_store_2[key_id] = prelabel2[num_i]
    except:
        pass
    try:
        pre_store_3 = dict()
        for num_i, key_id in enumerate(key_store_3.keys()):
            pre_store_3[key_id] = prelabel3[num_i]
    except:
        pass
    try:
        pre_store_4 = dict()
        for num_i, key_id in enumerate(key_store_4.keys()):
            pre_store_4[key_id] = prelabel4[num_i]
    except:
        pass
    all_pre_store = {**pre_store_1,**pre_store_2,**pre_store_3,**pre_store_4}




    #for key_id in all_pre_store.keys():
    #    print(np.frombuffer(key_id)[0], np.frombuffer(key_id)[1],np.frombuffer(key_id)[2],np.frombuffer(key_id)[3],np.frombuffer(key_id)[4])

    def extract_headers_from_fasta(fasta_file):
        with open(fasta_file, 'r') as file:
            headers = [record.id for record in SeqIO.parse(file, 'fasta')]
        return headers

    fasta_file = sys.argv[1]
    headers = extract_headers_from_fasta(fasta_file)



    sort_dict_pre = sorted(all_pre_store.keys(), key=lambda t: np.frombuffer(t)[0])

    all_count = count_1+count_2+count_3+count_4
    start = 0 
    x_ = []
    try:
        # Adjust the output file paths
        log_path = os.path.join(dir_name, 'log.tsv')
        plot_path = os.path.join(dir_name, 'Distribution_map.png')

        with open(log_path,'w+') as f1:

                  
            num_seq_effective = 0
            results = []
            while start<all_count:
                header = headers[num_seq_effective]

                step = int(np.frombuffer(sort_dict_pre[start])[4])
                denominator = 0
                weight = 0
                for i in sort_dict_pre[start:start+step]:
                    denominator += np.frombuffer(i)[2]
                    weight += np.frombuffer(i)[2]*all_pre_store[i]
                score = weight/denominator
                if score[0]>score[1]:
                    name = 'Virus'
                else:
                    name = 'non-Virus'
                results.append([header, score[0], score[1], name])
                num_seq_effective+=1

                x_.append(score[0])
                start +=step
            df = pd.DataFrame(results, columns=['Sequence_ID', 'Virus_Score', 'Non_Virus_Score', 'Taxon'])
            # Save the DataFrame to tsv file
            df.to_csv(f1, index=False, sep='\t')
            time_run = time.time() - start_time

        plt.hist(x_)
        plt.savefig(plot_path)
    except:
        pass
    print('Decontamination has been completed')
    sys.exit()

# If the filter argument is not 'yes', then continue with the existing program

start_time = time.time()
t = time.localtime()
current_time = time.strftime("%H_%M_%S_", t)
try:
    # Extract the file name from the provided path
    file_local_fasta = os.path.basename(sys.argv[1])
    
    # Generate a new directory name using the current time and file name
    dir_name = str(current_time) + file_local_fasta
    
    # Make the directory
    os.makedirs(dir_name)

except FileExistsError:
    print(f"Directory {dir_name} already exists!")
except Exception as e:
    print(f"An error occurred: {e}")
    
    
def getKmers(sequence, size):
    return [sequence[x:x+size] for x in range(len(sequence) - size + 1)]

def sgtfunc(filename,merlen=3):
    listapl=["AAA", "AAC", "AAG", "AAT",
"ACA", "ACC", "ACG", "ACT",
"AGA", "AGC", "AGG", "AGT",
"ATA", "ATC", "ATG", "ATT",
"CAA", "CAC", "CAG", "CAT",
"CCA", "CCC", "CCG", "CCT",
"CGA", "CGC", "CGG", "CGT",
"CTA", "CTC", "CTG", "CTT",
"GAA", "GAC", "GAG", "GAT",
"GCA", "GCC", "GCG", "GCT",
"GGA", "GGC", "GGG", "GGT",
"GTA", "GTC", "GTG", "GTT",
"TAA", "TAC", "TAG", "TAT",
"TCA", "TCC", "TCG", "TCT",
"TGA", "TGC", "TGG", "TGT",
"TTA", "TTC", "TTG", "TTT"]
    dictuncertain = {
        'N':'G',
        'X':'A',
        'H':'T',
        'M':'C',
        'K':'G',
        'D':'A',
        'R':'G',
        'Y':'T',
        'S':'C',
        'W':'A',
        'B':'C',
        'V':'G',
    }

    num = 0


    with open(filename,'r') as handle:


        key_store = dict()
        count_1=0
        count_2=0
        count_3=0
        count_4=0
        
        for seq_id, rec in enumerate(SeqIO.parse(handle, 'fasta')):
            address1=str(rec.seq)



            for word1, word2 in dictuncertain.items():



                address1 = address1.replace(word1, word2)


            sequence_id = np.zeros(5)
            sequence_id[0]=seq_id
            _length_sequence = len(address1)

            if 2>len(address1[_length_sequence//1800*1800:]):
                sequence_id[4]=_length_sequence//1800
            else:
                sequence_id[4]=_length_sequence//1800+1
            for _num_i in range(_length_sequence//1800+1):
                
                sequence_id[1]=_num_i
                if len(address1[_num_i*1800 : _num_i*1800+1800])<2:
                    pass

                elif 2<=len(address1[_num_i*1800 : _num_i*1800+1800])<400:
                    count_1+=1
                    sequence_id[2]=len(address1[_num_i*1800 : _num_i*1800+1800])
                    sequence_id[3]=1

                    key_store[sequence_id.tobytes()]=[count_1, getKmers(address1[_num_i*1800 : _num_i*1800+1800],merlen)]
                elif 400<=len(address1[_num_i*1800 : _num_i*1800+1800])<800:
                    count_2+=1
                    sequence_id[2]=len(address1[_num_i*1800 : _num_i*1800+1800])
                    sequence_id[3]=2
                    key_store[sequence_id.tobytes()]=[count_2, getKmers(address1[_num_i*1800 : _num_i*1800+1800],merlen)]
                elif 800<=len(address1[_num_i*1800 : _num_i*1800+1800])<1200:
                    count_3+=1
                    sequence_id[2]=len(address1[_num_i*1800 : _num_i*1800+1800])
                    sequence_id[3]=3
                    key_store[sequence_id.tobytes()]=[count_3, getKmers(address1[_num_i*1800 : _num_i*1800+1800],merlen)]
                elif 1200<=len(address1[_num_i*1800 : _num_i*1800+1800])<=1800:
                    count_4+=1
                    sequence_id[2]=len(address1[_num_i*1800 : _num_i*1800+1800])
                    sequence_id[3]=4
                    key_store[sequence_id.tobytes()]=[count_4, getKmers(address1[_num_i*1800 : _num_i*1800+1800],merlen)]
        
        global key_store_1 
        global key_store_2 
        global key_store_3 
        global key_store_4 
        global data_set_1
        global data_set_2
        global data_set_3
        global data_set_4
        key_store_1 = dict()
        key_store_2 = dict()
        key_store_3 = dict()
        key_store_4 = dict()
        for key_id in key_store.keys():

            if np.frombuffer(key_id)[3]==1:
                key_store_1[key_id]=key_store[key_id]
            if np.frombuffer(key_id)[3]==2:
                key_store_2[key_id]=key_store[key_id]
            if np.frombuffer(key_id)[3]==3:
                key_store_3[key_id]=key_store[key_id]
            if np.frombuffer(key_id)[3]==4:
                key_store_4[key_id]=key_store[key_id]


    sgt = SGT(alphabets=listapl,kappa=1, 
              flatten=True, 
              lengthsensitive=False, 
              mode='multiprocessing')            



    try:
        corpus = pd.DataFrame(key_store_1.values(),columns=['id', 'sequence'])
        sgtpd = sgt.fit_transform(corpus)
        sgtpd.pop('id')
        data_set_1 = tf.data.Dataset.from_tensor_slices((sgtpd.values))
    except:
        pass
    try:
        corpus = pd.DataFrame(key_store_2.values(),columns=['id', 'sequence'])
        sgtpd = sgt.fit_transform(corpus)

        sgtpd.pop('id')
        data_set_2 = tf.data.Dataset.from_tensor_slices((sgtpd.values))
    except:
        pass
    try:
        corpus = pd.DataFrame(key_store_3.values(),columns=['id', 'sequence'])
        sgtpd = sgt.fit_transform(corpus)

        sgtpd.pop('id')
        data_set_3 = tf.data.Dataset.from_tensor_slices((sgtpd.values))
    except:
        pass
    try:
        corpus = pd.DataFrame(key_store_4.values(),columns=['id', 'sequence'])
        sgtpd = sgt.fit_transform(corpus)

        sgtpd.pop('id')
        data_set_4 = tf.data.Dataset.from_tensor_slices((sgtpd.values))
    except:
        pass
    return (count_1, count_2, count_3, count_4)


##########################################################################

##########################################################################

def input_sample(input_dataset, repeat_count=1, batch_size=1, cpus=20):
    """Parses tfrecord and returns dataset for training/eval.

    Args:
        input_tfrec: Filenames of tfrecord.
        repeat_count: Number of epochs.
        batch_size: Number of examples returned per iteration.
        cpus: Number of cores used to running input pipeline.
        max_len: Length of the longest sequence.

    Returns:
        Dataset of (reads, label) pairs.
    """
    def _parse_4function(serialized):
        read = tf.reshape(serialized,[64, 64,1])
        return read
    dataset = input_dataset

    dataset = dataset.repeat(repeat_count)
    dataset = dataset.map(map_func=_parse_4function, num_parallel_calls=cpus)
    dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes=(
                                       tf.TensorShape([None,None,None])))

    return dataset
####################################################################

(count_1, count_2, count_3, count_4)=sgtfunc(str(sys.argv[1]))

try:
    model = tf.keras.models.load_model('./core/1.hdf5')
    input_sequence1 =data_set_1
    prelabel1 = model.predict(input_sample(input_sequence1), steps=count_1)
except:
    pass

try:
    model = tf.keras.models.load_model('./core/2.hdf5')

    input_sequence2 =data_set_2
    prelabel2 = model.predict(input_sample(input_sequence2), steps=count_2)

except:
    pass
try:
    model = tf.keras.models.load_model('./core/3.hdf5')

    input_sequence3 =data_set_3
    prelabel3 = model.predict(input_sample(input_sequence3), steps=count_3)
except:
    pass
try:

    model = tf.keras.models.load_model('./core/4.hdf5')

    input_sequence4 =data_set_4
    prelabel4 = model.predict(input_sample(input_sequence4), steps=count_4)

except:
    pass


try:
    pre_store_1 = dict()
    for num_i, key_id in enumerate(key_store_1.keys()):
        pre_store_1[key_id] = prelabel1[num_i]
except:
    pass
try:
    pre_store_2 = dict()
    for num_i, key_id in enumerate(key_store_2.keys()):
        pre_store_2[key_id] = prelabel2[num_i]
except:
    pass
try:
    pre_store_3 = dict()
    for num_i, key_id in enumerate(key_store_3.keys()):
        pre_store_3[key_id] = prelabel3[num_i]
except:
    pass
try:
    pre_store_4 = dict()
    for num_i, key_id in enumerate(key_store_4.keys()):
        pre_store_4[key_id] = prelabel4[num_i]
except:
    pass
all_pre_store = {**pre_store_1,**pre_store_2,**pre_store_3,**pre_store_4}




#for key_id in all_pre_store.keys():
#    print(np.frombuffer(key_id)[0], np.frombuffer(key_id)[1],np.frombuffer(key_id)[2],np.frombuffer(key_id)[3],np.frombuffer(key_id)[4])

def extract_headers_from_fasta(fasta_file):
    with open(fasta_file, 'r') as file:
        headers = [record.id for record in SeqIO.parse(file, 'fasta')]
    return headers

fasta_file = sys.argv[1]
headers = extract_headers_from_fasta(fasta_file)



sort_dict_pre = sorted(all_pre_store.keys(), key=lambda t: np.frombuffer(t)[0])

all_count = count_1+count_2+count_3+count_4
start = 0 
x_ = []
try:
    # Adjust the output file paths
    log_path = os.path.join(dir_name, 'log.tsv')
    plot_path = os.path.join(dir_name, 'Distribution_map.png')

    with open(log_path,'w+') as f1:

        print('++++++++++++++++++++++++++++++IPEV++++++++++++++++++++++++++++++\n')

        print('Prokaryotes Viruses score||Eukaryotes  Viruses score \n')
        num_seq_effective = 0
        results = []
        while start<all_count:
            header = headers[num_seq_effective]

            step = int(np.frombuffer(sort_dict_pre[start])[4])
            denominator = 0
            weight = 0
            for i in sort_dict_pre[start:start+step]:
                denominator += np.frombuffer(i)[2]
                weight += np.frombuffer(i)[2]*all_pre_store[i]
            score = weight/denominator
            if score[0]>score[1]:
                name = 'Prokaryotes Virus'
            else:
                name = 'Eukaryotes  Virus'
            print('Seq_id:{0:^12}=====>   {1:.4f},{2:.4f}|| VirusType: {3}  '.format(np.frombuffer(sort_dict_pre[start])[0],score[0],score[1],name))
            results.append([header, score[0], score[1], name])
            num_seq_effective+=1

            x_.append(score[0])
            start +=step
        df = pd.DataFrame(results, columns=['Sequence_ID', 'Prokaryotic_Virus_Score', 'Eukaryotic_Virus_Score', 'Virus_Taxon'])
        # Save the DataFrame to tsv file
        df.to_csv(f1, index=False, sep='\t')
        time_run = time.time() - start_time
        print('Total Seq numbers are ==> {0}\n'.format(num_seq_effective))
        print("IPEV Run Time --- %.4f seconds ---" % (time_run))
    plt.hist(x_)
    plt.savefig(plot_path)
except:
    pass
