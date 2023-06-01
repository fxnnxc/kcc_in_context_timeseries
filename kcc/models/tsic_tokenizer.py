from abc import ABC, abstractmethod
from typing import Dict, Union

import os
import numpy as np
import pandas as pd
import pickle
import einops
from jaxtyping import Float, Array, Integer
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import TimeSeriesKMeans, KShape, KernelKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
from fire import Fire

class Tokenizer(ABC):
    
    @abstractmethod
    def get_prototype_id(self, x: Float[Array, "batch_size seq_len dim"]) -> Integer[Array, "batch_size"]:
        """From a patch of time series, get the best propto"""
        
    @abstractmethod
    def all_prototypes(self) -> Dict[int, Float[Array, "seq_len dim"]]:
        """Get all the prototypes"""
        
    @abstractmethod
    def get_prototype(self, id: int) -> Float[Array, "seq_len dim"]:
        """Get prototype given ID"""
        
        
class TokenizerFromCluster(Tokenizer):
    
    prototypes: Dict[int, Array]
    mean_var_scaler: TimeSeriesScalerMeanVariance
    resample_scaler: TimeSeriesResampler
    clustering_model: Union[TimeSeriesKMeans, KShape, KernelKMeans]
    is_trained: bool = False

    
    def __init__(self,
                 num_prototypes: int, 
                 seq_len: int, 
                 cluster_algo: str = "kmean", 
                 verbose=False):
        self.num_prototypes = num_prototypes
        self.seq_len = seq_len
        self.cluster_algo = cluster_algo
        
        # rescale data again for each patch
        self.mean_var_scaler = TimeSeriesScalerMeanVariance()
        # resample if nescessary. This makes data become smoother
        self.resample_scaler = TimeSeriesResampler(seq_len)
        if cluster_algo == "kmean":
            self.clustering_model = TimeSeriesKMeans(n_clusters=self.num_prototypes,
                                                metric="dtw",
                                                n_jobs=-1,
                                                n_init=3,
                                                verbose=verbose)
        elif cluster_algo == "kshape":
            # K-shape seems to run faster 
            # see more about K-shape in this:
            # https://sigmodrecord.org/publications/sigmodRecord/1603/pdfs/18_kShape_RH_Paparrizos.pdf
            self.clustering_model = KShape(n_clusters=num_prototypes,
                                           n_init=3,
                                           verbose=verbose)
        elif cluster_algo == "kernelkmean":
            self.clustering_model = KernelKMeans(n_clusters=num_prototypes,
                                                 n_jobs=-1,
                                                 n_init=3,
                                                 verbose=verbose)
        else:
            raise ValueError(f"Unknown clustering algorithm: {cluster_algo}. Please use either `kshape` or `kmean` or `kernelkmean`")
            
    
    def train(self, data: Float[Array, "num_data seq_len dim"]):
        data = self.preprocess(data)
        self.clustering_model.fit(data)
        self.is_trained = True
        if isinstance(self.clustering_model, KernelKMeans):
            # `KernelKMeans` does not need centroids this is not important
            cluster_ids = self.clustering_model.predict(data)
            self.prototypes = {}
            for i in range(self.num_prototypes):
                prototype = np.mean(data[cluster_ids==i], axis=0)
                self.prototypes[i] = prototype
        else:
            cluster_center = self.clustering_model.cluster_centers_
            self.prototypes = dict((key, value) for key, value in enumerate(cluster_center))
            
    
    def get_prototype_id(self, x: Array) -> int:
        if not self.is_trained:
            raise RuntimeError("Clutering algorithm is not trained yet")
        x = self.preprocess(x)
        id = self.clustering_model.predict(x)    
        return id
    
    def get_prototype(self, id: int) -> Array:
        if not self.is_trained:
            raise RuntimeError("Clutering algorithm is not trained yet")
        if id >= self.num_prototypes:
            raise ValueError(f"Index error! There are {self.num_prototypes} prototypes. But input ID is {id}")
        return self.prototypes[id]
    
    def all_prototypes(self) -> Dict[int, Array]:
        if not self.is_trained:
            raise RuntimeError("Clutering algorithm is not trained yet")
        return self.prototypes
    
    def preprocess(self, x):
        x = self.mean_var_scaler.fit_transform(x)
        x = self.resample_scaler.fit_transform(x)
        return x
    
    
def save_tokenizer(tokenizer: Tokenizer, save_file: str):
    """Save tokenizer as a file"""
    with open(save_file, "wb") as f:
        pickle.dump(tokenizer, f, pickle.HIGHEST_PROTOCOL)
    print(f"Save tokenizer to file: {save_file}")
    

def load_tokenizer(save_file: str) -> Tokenizer:
    """Load tokenizer as a file"""
    with open(save_file, "rb") as f:
        tokenizer = pickle.load(f)
    print(f"Load tokenizer from file: {save_file}")
    return tokenizer


def tokenize(root_path: str = "./data/electricity",
             data_file: str = "electricity.csv",
             num_prototypes: int = 25, 
             patch_len: int = 32, 
             clustering_seq_len: int = 16, 
             clustering_algo: str = "kmean",
             tokenizer_file: str = None,
             ):
    """Tokenize function
    Finally, it will save prototype IDs, mean, standard deviation, and patched data to file.
    If tokernizer is trained for the first time, tokenizer is save to file as well
    
    """
    
    if tokenizer_file is not None:
        tokenizer = load_tokenizer(os.path.join(root_path, tokenizer_file))
        assert tokenizer.num_prototypes == num_prototypes
        assert tokenizer.seq_len == clustering_seq_len
        assert tokenizer.cluster_algo == clustering_algo
    else:
        tokenizer = TokenizerFromCluster(num_prototypes=num_prototypes,
                                         seq_len=clustering_seq_len,
                                         cluster_algo=clustering_algo,)
        
    df = pd.read_csv(os.path.join(root_path, data_file))
    # consider univariate time series ONLY
    raw_data = df[["OT"]].values
    scaler = StandardScaler()
    scaler.fit(raw_data)
    data = scaler.transform(raw_data)
    
    total_seq_len = data.shape[0]
    num_patches = total_seq_len // patch_len
    
    data = einops.rearrange(data[:num_patches * patch_len, ...],
                            "(num_patches patch_len) dim -> num_patches patch_len dim",
                            num_patches=num_patches)
    
    if not tokenizer_file:
        print(f"Train clustering algorithm: {clustering_algo}")
        tokenizer.train(data)
        print("Finish training!")
    else:
        print("No training required")
        
    prototypes_ids = tokenizer.get_prototype_id(data)
    mean = np.mean(data, axis=1)
    std = np.std(data, axis=1)
    
    save_file = os.path.join(root_path, 
                             f"processed_data_{num_prototypes}_len_{clustering_seq_len}_algo_{clustering_algo}.pkl")
    with open(save_file, "wb") as f:
        pickle.dump([prototypes_ids, mean, std, data], f)
    print(f"Save prototype IDs, mean, var, and data to file: {save_file}")
    
    if tokenizer_file is None:
        tokenizer_name = f"tokenizer_proto_{num_prototypes}_len_{clustering_seq_len}_algo_{clustering_algo}.pkl"
        tokenizer_file = os.path.join(root_path, tokenizer_name)
        save_tokenizer(tokenizer, tokenizer_file)
    
if __name__ == "__main__":
    
    Fire(tokenize)