from pythainlp import word_tokenize
from pythainlp.spell import correct
import json
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
    def predict(self,sentense):
        list_word = word_tokenize(sentense,engine=self.engine)
        edit_list = []
        start_index = 0
        for raw in list_word:
            correct_word = correct(raw)
            if correct_word != raw:
                id = raw+"_"+str(start_index)
                edit_list.append({
                    "id":id,
                    "start_index":start_index,
                    "end_index":start_index+len(raw),
                    "word":raw,
                    "edit":correct_word})
            start_index += len(raw)
        return edit_list