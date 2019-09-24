import numpy as np
from scipy.special import gammaln

class SegDurPrior(object):
    def __init__(self, config, data):
        self.hyper_params = None
        self.segmentation_log_prior = None
        self.dataset_len = data.n_sents
        self.n_docs = 1 #Legacy code
        self.docs_hyper_params = [[] for i in range(self.n_docs)]
        
        prior_dist = config["seg_dur_prior_config"][0]
        hyper_params_raw = config["seg_dur_prior_config"][1]
        self.docs_hyper_params = [hyper_params_raw]
        self.docs_hyper_params = self.unpack_hyper_params(self.docs_hyper_params)
        
        if prior_dist == "normal":
            self.segmentation_log_prior = self.segmentation_normal_log_prior
        elif prior_dist == "beta_bern":
            self.segmentation_log_prior = self.segmentation_beta_bern_log_prior
        elif prior_dist == "gamma_poisson":
            self.segmentation_log_prior = self.segmentation_gamma_poisson_log_prior
            
    def unpack_hyper_params(self, docs_hyper_params):
        unpacked_params = [[] for i in range(len(docs_hyper_params[0]))]
        for hp in docs_hyper_params:
            for i, val in enumerate(hp):
                unpacked_params[i].append(val)
        return np.array(unpacked_params)
                  
    def normal_log_prior(self, seg_size, doc_i):
        mean = self.docs_hyper_params[0][doc_i]
        std = self.docs_hyper_params[1][doc_i]
        norm_logpdf = -np.log((np.sqrt(2*np.pi*(std**2))))-(seg_size-mean)**2/(2*(std**2))
        return norm_logpdf
        
    def segmentation_normal_log_prior(self, u_clusters):
        log_prior = 0.0
        for u_cluster in u_clusters:
            for doc_i in u_cluster.get_docs():
                u_begin, u_end = u_cluster.get_segment(doc_i)
                seg_size = u_end-u_begin+1
                log_prior += self.normal_log_prior(seg_size, doc_i)
        return log_prior
    
    def segmentation_beta_bern_log_prior(self, u_clusters):
        f1 = np.zeros(self.n_docs)
        f2 = np.zeros(self.n_docs)
        denom = np.zeros(self.n_docs)
        for u_cluster in u_clusters:
            for doc_i in u_cluster.get_docs():
                u_begin, u_end = u_cluster.get_segment(doc_i)
                seg_len = u_end-u_begin+1
                f1[doc_i] += 1.0
                f2[doc_i] += seg_len
                denom[doc_i] += seg_len
        
        alpha = self.docs_hyper_params[0]
        beta = self.docs_hyper_params[1]
        f2 -= f1
        f1 += alpha
        f2 += beta
        denom += alpha+beta
        log_prior = np.sum(gammaln(f1)+gammaln(f2)-gammaln(denom))
        return log_prior
    
    def segmentation_gamma_poisson_log_prior(self, u_clusters):
        n_rho1 = np.zeros(self.n_docs)
        n = np.zeros(self.n_docs)
        for u_cluster in u_clusters:
            for doc_i in u_cluster.get_docs():
                u_begin, u_end = u_cluster.get_segment(doc_i)
                seg_len = u_end-u_begin+1
                n_rho1[doc_i] += 1.0
                n[doc_i] += seg_len
                
        alpha = self.docs_hyper_params[0]
        beta = self.docs_hyper_params[1]
        f1 = gammaln(n_rho1+alpha)
        f2 = (n_rho1+alpha)*np.log(n+beta)
        log_prior = np.sum(f1-f2)
        return log_prior
                