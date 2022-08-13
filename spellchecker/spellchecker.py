from pythainlp import word_tokenize
from pythainlp.spell import correct
from pythainlp import sent_tokenize, word_tokenize
from pythainlp import thai_digits, thai_letters
from pythainlp.spell import NorvigSpellChecker
from pythainlp.corpus import download , get_corpus_path , get_corpus
import numpy as np
import json
class Corpus():
  
  def __init__(self,dict_word_fre):
    self.dictt = dict_word_fre
    self.word = list(dict_word_fre.keys())
    self.fre = list(dict_word_fre.values())

  def __len__(self):
    return len(self.word)

  def __getitem__(self,word):
    return self.dictt[word]

class Corpus_from_dow(Corpus):
  def __init__(self,name):
    dict_cor = self.dow(name)
    Corpus.__init__(self,dict_cor)
  def dow(self,name):
    download(name)
    path = get_corpus_path(name)
    cor = get_corpus(path)
    word = ["".join(x.split("\t")[:-1]) for x in cor]
    fre = [int(x.split("\t")[-1]) for x in cor]
    dict_cor = dict(zip(word, fre))
    return dict_cor

class candidater():
  def edits1(self,word):
    "All edits that are one edit away from `word`."    
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    # transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    #replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    # inserts    = [L + c + R               for L, R in splits for c in letters]   
    return set(deletes)#set(deletes + transposes + replaces + inserts)

  def delet2dict(self,dict_new,text):   
    for delet1 in self.edits1(text):
      if dict_new.get(delet1) is None:
        dict_new.update({
            delet1:{text}
            })
      else:
        dict_new.get(delet1).add(text)

  def __init__(self,unigrame):
    self.unidict = unigrame.dictt
    uniword = unigrame.word    
    dict_new = {}
    for text in uniword:     
      self.delet2dict(dict_new,text)
    self.dict_new = dict_new

  def candidate(self,word):
    set_candidate = set()

    if word in self.dict_new.keys():
      set_candidate.update(self.dict_new[word])
      
    del_set = self.edits1(word)

    for word_del in del_set:
      
      if word_del in self.unidict.keys():
        set_candidate.add(word_del)

      if word_del in self.dict_new.keys():
        set_candidate.update(self.dict_new[word_del])
    return set_candidate


class spell_checker():
    def __init__(self,config_path) :
        try :
            with open(config_path) as fp:
                self.config  = json.load(fp)
                self.name = self.config['name']
            print(f"[LOG] - successfully loaded {self.name}")

        except :
            self.config = {
            "name" : "POS_Model_Default_Configuration",
            "tokenizer" : "deepcut",
            }
            self.name = self.config['name']
            print("[WARN] - Load Error , rolling back to default configuration")
        self.engine = self.config["tokenizer"]
        checker = NorvigSpellChecker()
        unigrame = Corpus(dict(checker.dictionary()))
        trigrame = Corpus_from_dow("tnc_trigram_word_freqs")
        bigrame = Corpus_from_dow("tnc_bigram_word_freqs")
        candit = candidater(unigrame)
        #combine function for find correct words
        def correct_word_s(input_words):  
            #2. find possible word for each word
            word__wrong_list = input_words

            record_possible_word =[]
            
            
            for word in word__wrong_list:
                # each_word_list =[]
                each_word_list = candit.candidate(word)
                record_possible_word.append(each_word_list)  

            #3.combine each possible words to trigram word

            if len(record_possible_word) != 0:
                all_edit_words =[]
                for i in range(len(word__wrong_list)):
                    for item in record_possible_word[i]:
                        if i == 0:
                            new_word = item + word__wrong_list[1] + word__wrong_list[2]
                        elif i == 1:
                            new_word = word__wrong_list[0] + item + word__wrong_list[2]
                        else:
                            new_word = word__wrong_list[0] + word__wrong_list[1] + item
                        all_edit_words.append(new_word)
                    
            else:
                print('no record possible word')

            #4. loop edit word in dict again to find most freq(prob)

            selected_words=[]
            freq_words=[]


            for item in all_edit_words:    
                try:
                    freq_word = trigrame.dictt[item]
                    selected_words.append(item)
                    freq_words.append(int(freq_word))

                except:
                    pass
            #after augment may be reduce to / words (bigrame dict)
            if len(selected_words) == 0:
                for item in all_edit_words:  
                    try:
                        freq_word = bigrame.dictt[item]
                        selected_words.append(item)
                        freq_words.append(int(freq_word))
                
                    except:
                        pass

            #after augment may be reduce to 1 words (unigrame dict)
            if len(selected_words) == 0:
                for item in all_edit_words:  
                    if item in unigrame.word:
                        selected_words.append(item)
                        freq_words.append(unigrame[unigrame.df['words'] == item].freq)
                

            #select the most freq
            try:
                correct_word = selected_words[np.argmax(freq_words)]
                return correct_word
            except: #may be correct but wrong cut that why we cannot find it in dict   
                return "".join(input_words)
        N = sum(trigrame.dictt.values())

        def is_overlapped(a,b):
            if a[0] > b[0]: a,b = b,a
            if a[1] > b[0]: return True
            return False

        def add_prop(sentence,candidates):
            sentence = list(sentence)
            window_size = 3
            result = []
            for candidate in candidates:
                start, end = candidate["start"], candidate["end"]
                sen = sentence[:]
                sen[start:end] = list(candidate["new_word"])
                segs = word_tokenize(''.join(sen), engine='attacut')
                prod = 1
                for i in range(len(segs) - window_size+1):
                    gram = segs[i:i+window_size]
                    gram = ''.join(gram)
                    if gram in trigrame.dictt:
                        ele = trigrame.dictt[gram]
                    else:
                        ele = 0
                    prod += ele
                    candidate.update({"prop":prod})
                    result.append(candidate)
            return result
                    
        def remove_overlap(sentence, candidates):
            candidates_prop = add_prop(sentence,candidates)
            # candi = sorted(candidates_prop, key=lambda x: x['prop'])
            candi = candidates_prop
            sen = sentence[:]
            i = 0
            list_candidate = []
            while i < len(candi):
                a,b,c = None,None,None
                a = [candi[i]["start"], candi[i]["end"]]
                if i + 1 < len(candi):
                    b = [candi[i+1]["start"], candi[i+1]["end"]]
                if i + 2 < len(candi):
                    c = [candi[i+2]["start"], candi[i+2]["end"]]
                
                if b is None and c is None:
                    list_candidate.append(candi[i])
                    break
                
                if b is not None and c is None:
                    if is_overlapped(a,b):
                        if candi[i]['prop'] >= candi[i+1]['prop']:
                            list_candidate.append(candi[i])
                        else:
                            list_candidate.append(candi[i+1])
                    else:
                        list_candidate.append(candi[i])
                        list_candidate.append(candi[i+1])
                    break

            else:
                if is_overlapped(a,b):
                    if is_overlapped(a,c):
                        agmax = np.argmax([candi[i]['prop'], candi[i+1]['prop'], candi[i+2]['prop']])
                        list_candidate.append(candi[agmax])
                        i += 3
                    else:
                        if candi[i]['prop'] >= candi[i+1]['prop']:
                            list_candidate.append(candi[i])
                        else:
                            list_candidate.append(candi[i+1])
                        i += 2
                else:
                    list_candidate.append(candi[i])
                    list_candidate.append(candi[i+1])
                    list_candidate.append(candi[i+2])
                    i += 3

            def get_prob(list_dict):
                return list_dict["prop"]
            
            ans_list = list()
            count_idx = 0
            list_candidate.sort(reverse=True,key=get_prob)
            return list_candidate
        def Pmick(paragraph):
            def prepare_index(ans_list, length_acc_list):
                for idx, sen in enumerate(ans_list):
                    if idx != 0:
                        for each in sen:
                            each['start']  += length_acc_list[idx-1]
                            each['end']  += length_acc_list[idx-1]
                    
                return ans_list
            
            ans_list = list()
            count_idx = 0

            # Find accumulative index in pg
            sentence_list = sent_tokenize(paragraph)
            length_acc_list = list()
            length_acc = 0
            for i in sentence_list:
                length_acc += len(i)
                length_acc_list.append(length_acc)


            for text in sentence_list:
                
                word_deepcut =  word_tokenize(text, engine="deepcut") #deep or atta

                full_word=[]
                Wrong_word=[]

                #index
                start_idx_list=[]
                end_idx_list=[]

                #step1. 
                count = 1

                for idx in range(len(word_deepcut)):

                    if idx == 0: #only first index to add </s> for first 2 words
                        word=word_deepcut[idx:idx+2]

                        start_idx_list.append(0) #add value 0
                        count_word=''.join(word)    
                        end_idx_list.append(len(count_word))   #add start+len  

                        word.insert(0,'<s/>')
                        full_word.append(word)

                    
                    if idx == len(word_deepcut)-2:
                        word=word_deepcut[idx:idx+2]
                        count_word=''.join(word)    
                        
                        start_idx_list.append(count-1)
                        count+=len(count_word)
                        end_idx_list.append(count-1)
                        word.append('<s/>')
                        full_word.append(word)
                        break

                    else:
                        word=word_deepcut[idx:idx+3]
                        count_word=''.join(word)    
                        start_idx_list.append(count-1)    
                        end_idx_list.append(count+len(count_word)-1)
                        count+=len(word[0]) #next start index

                    full_word.append(word)
                    start_idx_list[1] = 0 #direct apply


                #step2.
                wrong_words =[]
                correct_words =[]
                wrong_word_combine=[]
                selected_start_index=[]
                selected_end_index=[]

                for i in range(len(full_word)):
                    tri_word="".join(full_word[i])
                    if tri_word in trigrame.word:
                        # correct_words.append(item)
                        pass   
                    else:    
                        wrong_words.append(full_word[i])
                        wrong_word_combine.append(tri_word)
                        selected_start_index.append(start_idx_list[i])
                        selected_end_index.append(end_idx_list[i])


                edit_word_list=[]

                #find correct word
                for item in wrong_words:
                    correct_one = correct_word_s(item)
                    edit_word_list.append(correct_one)

                #find dict list
                result =[]
                for i in range(len(edit_word_list)):
                    edit_word ={}
                    if wrong_word_combine[i] != edit_word_list[i]:    
                        edit_word["start"] = selected_start_index[i]
                        edit_word["end"] = selected_end_index[i]
                        edit_word["old_word"] = wrong_word_combine[i]
                        edit_word["new_word"] = edit_word_list[i]
                        result.append(edit_word)

                ans_list.append(result)
            ans_list = [remove_overlap(sentence_list[index],ans_list[index])[:2] for index in  range(len(ans_list))]
            ans = prepare_index(ans_list, length_acc_list)
            return ans
        self.Pmick = Pmick

    def predict(self,sentense):
        ans = self.Pmick(sentense)
        ture_ans = []
        for i in ans:
            for dict_ in i:
                dict_.pop("prop")
                ture_ans.append(dict_)
        return ture_ans