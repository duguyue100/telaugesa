"""Image Caption Generation Utility Functions

Some utilites is taken from [neuraltalk](https://github.com/karpathy/neuraltalk/)
"""

import json;
import scipy.io;
from collections import defaultdict;

class Im2TextDataProvider(object):
    """Provide Image Caption Generation Data"""
    
    def __init__(self,
                 feature_name,
                 description_name):
        """Init a data provider
        
        Parameters
        ----------
        feature_name : string
            file name of features
        description_name : string
            file name of description
        """
        
        self.feature_name=feature_name;
        self.description_name=description_name;
        
        self.initialize_database();
        
    def initialize_database(self):
        """
        Features are organized in (number of samples, dim of feature)
        
        """
    
        print "[MESSAGE] Loading the dataset"    
        self.feature=scipy.io.loadmat(self.feature_name)["feats"].T;
        self.desc_struct=json.load(open(self.description_name, 'r'));
        
        self.desc=defaultdict(list);
        for img in self.desc_struct["images"]:
            self.desc[img["split"]].append(img);
            
        print "[MESSAGE] The dataset is loaded"
    
    def getImage(self, imgdesc):
        """Get image feature from image description
        
        Parameters
        ----------
        imgdesc : structure
            an image description
        
        Returns
        -------
        feature : structure
            image feature
        """
        
        if not "feat" in imgdesc:
            imgdesc["feat"]=self.feature[imgdesc["imgid"]];
            
        return imgdesc;