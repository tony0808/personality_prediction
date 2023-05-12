import pandas as pd

class DataAccessManager:
    
    def __init__(self):
        self.abs_path = 'C:\\Users\\Admin\\Desktop\\ptuxiaki\\PersonalityTraitCode\\my_code\\data\\bigFive'
        self.traitMap = {
            'o':'openessData',
            'c':'consiousnessData',
            'e':'extraversionData',
            'a':'agreeablenessData',
            'n':'neurotismData'
        }

    def get_data(self, trait_initial):
        try:
            data = pd.read_csv(self.abs_path + '\\' + self.traitMap[trait_initial] + '.csv')
            return data
        except KeyError:
            print("Invalid parameter. Valid character parameters: 'o', 'c', 'e', 'a', 'n'")