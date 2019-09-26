import numpy as np
import copy
import operator
from tqdm import trange
import os
import seg_dur_prior as sdp
from scipy.special import gammaln
import collections

class BeamSeg():
    def __init__(self, data, seg_config):
        self.data = data
        self.beta = np.array([seg_config["beta"]]*data.W)
        self.prior_class = sdp.SegDurPrior(seg_config, self.data)
        self.desc = "BeamSeg"
        self.log_dir = "logs"
        self.W = data.W
        self.best_segmentation = [[] for i in range(self.data.n_sents)]
        self.seg_ll_C = gammaln(self.beta.sum())-gammaln(self.beta).sum()
        os.remove(self.log_dir+"dp_tracker_"+self.desc+".txt") if os.path.exists(self.log_dir+"dp_tracker_"+self.desc+".txt") else None
        
        
        self.max_cache = seg_config["max_cache"]
        self.log_flag = seg_config["log_flag"]
        
        self.u_order = [(u, 0) for u in range(self.data.n_sents)]
    
    def segment_ll(self, word_counts):
        '''
        Returns the likelihood if we considering all sentences (word_counts)
        as a single language model.
        :param seg_word_counts: vector with the size equal to the length of
        the vocabulary and values with the corresponding word counts.
        '''
        f1 = gammaln(word_counts+self.beta).sum()
        f2 = gammaln((word_counts+self.beta).sum())
        seg_ll = self.seg_ll_C+f1-f2
        return seg_ll
    
    def segmentation_ll(self, u_clusters):
        '''
        Returns the log likelihood of the segmentation of all documents.
        :param u_clusters: list of SentenceCluster corresponding to the best segmentation up to u-1
        '''
        segmentation_ll = 0.0
        for u_cluster in u_clusters:
            cached_ll = u_cluster.get_cluster_ll()
            if cached_ll is not None:
                cluster_ll = cached_ll
            else:
                word_counts = u_cluster.get_word_counts()
                cluster_ll = self.segment_ll(word_counts)
                u_cluster.set_cluster_ll(cluster_ll)
            
            segmentation_ll += cluster_ll
            
        segmentation_ll += self.prior_class.segmentation_log_prior(u_clusters)
                                
        return segmentation_ll
    
    def get_cluster_order(self, doc_i, u_clusters):
        cluster_k_list = []
        for u_cluster in u_clusters:
            if u_cluster.has_doc(doc_i):
                u_begin, u_end = u_cluster.get_segment(doc_i)
                cluster_k_list.append([u_cluster.k, u_begin])
        cluster_k_list = sorted(cluster_k_list, key=lambda x: x[1])
        ret_list = []
        for cluster_k in cluster_k_list:
            ret_list.append(cluster_k[0])
        return ret_list
    
    def get_k_cluster(self, k, u_clusters):
        for u_cluster in u_clusters:
            if u_cluster.k == k:
                return u_cluster
        return None
    
    def get_segmentation(self, doc_i, u_clusters):
        '''
        Returns the final segmentation for a document.
        This is done by backtracking the best segmentations
        in a bottom-up fashion.
        :param doc_i: document index
        '''
        if isinstance(u_clusters[0], collections.Iterable):
            u_clusters = u_clusters[0]
            
        hyp_seg = []
        cluster_order = self.get_cluster_order(doc_i, u_clusters)
        for k in cluster_order:
            u_cluster = self.get_k_cluster(k, u_clusters)
            u_begin, u_end = u_cluster.get_segment(doc_i)
            seg_len = u_end-u_begin+1
            doc_i_seg = list([0]*seg_len)
            doc_i_seg[-1] = 1
            hyp_seg += doc_i_seg
        return hyp_seg
    
    def get_final_segmentation(self, doc_i):
        u_clusters = self.best_segmentation[-1][0][1]
        hyp_seg = self.get_segmentation(doc_i, u_clusters)
        return hyp_seg
    
    def get_k_cluster_index(self, k, u_clusters):
        for i, u_cluster in enumerate(u_clusters):
            if u_cluster.k == k:
                return i
        return None
    
    def assign_target_k(self, u_begin, u_end, doc_i, k_target, u_clusters):
        i = self.get_k_cluster_index(k_target, u_clusters)
        if i is not None:
            u_k_target_cluster = u_clusters[i]
            u_k_target_cluster = copy.deepcopy(u_k_target_cluster)
            u_k_target_cluster.set_cluster_ll(None)
            u_clusters[i] = u_k_target_cluster
            u_k_target_cluster.add_sents(u_begin, u_end, doc_i)
        else:
            u_k_cluster = SentenceCluster(self.data, u_begin, u_end, [doc_i], k_target)
            u_clusters.append(u_k_cluster)
        return u_clusters
    
    def compute_seg_ll_seq(self, cached_segs, doc_i, u):
        '''
        Computes in sequentially the segmentation likelihood of assigning u to
        some topic k starting from a segmentation in cached_segs
        :param cached_segs: u_clusters for which we want to know the likelihood
        :param doc_i: document index from which u comes
        :param u: utterance index
        '''
        doc_i_segs = []
        for cached_seg in cached_segs:
            cached_u_clusters = cached_seg[1]
            if len(cached_seg[1]) == 0:
                test_clusters = [0]
            else:
                prev_cluster_k = cached_u_clusters[-1].k
                test_clusters = [prev_cluster_k, prev_cluster_k+1] #trying to add the paragraph to previous document or a possible new document
            for k in test_clusters:
                current_u_clusters = copy.copy(cached_u_clusters)
                current_u_clusters = self.assign_target_k(u, u, doc_i, k, current_u_clusters)
                seg_ll = self.segmentation_ll(current_u_clusters)
                doc_i_segs.append((seg_ll, current_u_clusters, k))
        return doc_i_segs
    
    def remove_seg_dups(self, doc_i_segs):
        '''
        Removes duplicate segmentations based on equal segmentation likelyhoods.
        Ensures that the cache does not end up with unecessary duplicates.
        :param doc_i_segs: list of tuples in the format (seg_ll, current_u_clusters, phi_tt, k)
        '''
        doc_i_segs = sorted(doc_i_segs, key=operator.itemgetter(0), reverse=True)
        no_dups_doc_i_segs = []
        prev_seg_ll = -np.inf
        for seg_result in doc_i_segs:
            seg_ll = seg_result[0]
            seg_clusters = seg_result[1]
            if seg_ll != prev_seg_ll:
                no_dups_doc_i_segs.append(seg_result)
            else:
                print("FOUND SEG DUP")
            prev_seg_ll = seg_ll
        return no_dups_doc_i_segs
    
    def cache_prune(self, cached_segs):
        return cached_segs[:self.max_cache]
        
    def greedy_segmentation_step(self, u_order=None):
        with open(self.log_dir+"dp_tracker_"+self.desc+".txt", "a+") as f:
            t = trange(len(u_order), desc='', leave=True)
            cached_segs = [(-np.inf, [])]
            for i in t:
                u = u_order[i][0]
                doc_i = u_order[i][1]
                t.set_description("(%d, %d)" % (u, doc_i))
                doc_i_segs = self.compute_seg_ll_seq(cached_segs, doc_i, u)
                        
                #no_dups_doc_i_segs = self.remove_seg_dups(doc_i_segs)
                no_dups_doc_i_segs = sorted(doc_i_segs, key=operator.itemgetter(0), reverse=True) #TODO: check that in this setup I dont get dups
                cached_segs = self.cache_prune(no_dups_doc_i_segs)
                
                if self.log_flag:
                    for i, cached_seg in enumerate(cached_segs):
                        f.write("(%d %d)\t%d\tll: %.3f\n"%(u, doc_i, i, cached_seg[0]))
                        for doc_j in range(self.data.n_docs):
                            f.write(str(self.get_segmentation(doc_j, cached_seg[1]))+"\n")
                        f.write("\n")
                    f.write("===============\n")
        cached_segs = sorted(cached_segs, key=operator.itemgetter(0), reverse=True)
        self.best_segmentation[-1] = cached_segs
        
    def segment_docs(self):
        self.greedy_segmentation_step(self.u_order)
        
class SentenceCluster(object):
    '''
    Class to keep track of a set of sentences (possibly from different documents)
    that belong to the same segment.
    '''
    def __init__(self, data, u_begin, u_end, docs, k, track_words=False):
        self.data = data
        self.k = k
        self.doc_segs_dict = {}
        #self.word_counts = np.zeros(data.W)
        self.track_words = track_words
        self.cluster_ll = None
        
        for doc_i in docs:
            doc_i_len = self.data.n_sents
            #Accounting for documents with different lengths
            if u_begin > doc_i_len-1:
                continue
            
            if u_end > doc_i_len-1:
                u_end_true = doc_i_len-1
            else:
                u_end_true = u_end
            self.doc_segs_dict[doc_i] = [u_begin, u_end_true]
            
        
        self.wi_list = []
        for doc_i in docs:
            doc_i_len = self.data.n_sents
            #Accounting for documents with different lengths
            if u_begin > doc_i_len-1:
                continue
            if u_end > doc_i_len-1:
                true_u_end = doc_i_len-1
            else:
                true_u_end = u_end
            if self.track_words:
                for u in range(u_begin, true_u_end+1):
                    d_u_words = self.data.d_u_wi_indexes[doc_i][u]
                    self.wi_list += d_u_words
                    
    def has_doc(self, doc_i):
        return doc_i in self.doc_segs_dict.keys()
    
    def add_sents(self, u_begin, u_end, doc_i):
        doc_i_len = self.data.n_sents
        #Accounting for documents with different lengths
        if u_begin > doc_i_len-1:
            return
        if u_end > doc_i_len-1:
            u_end = doc_i_len-1
            
        if self.has_doc(doc_i):
            current_u_begin, current_u_end = self.get_segment(doc_i)
            if u_begin < current_u_begin:
                current_u_begin = u_begin
            if u_end > current_u_end:
                current_u_end = u_end
            self.doc_segs_dict[doc_i] = [current_u_begin, current_u_end]
        else:
            self.doc_segs_dict[doc_i] = [u_begin, u_end]
            
        seg = list(range(u_begin, u_end+1))
        
        if self.track_words:
            for u in seg:
                d_u_words = self.data.d_u_wi_indexes[doc_i][u]
                self.wi_list += d_u_words                                                           
            
    def get_docs(self):
        return self.doc_segs_dict.keys()
        
    def get_word_counts(self):
        u_indexes = []
        for doc_i in self.get_docs():
            if doc_i > 0:
                doc_carry = self.data.docs_index[doc_i-1]
            else:
                doc_carry = 0
            u_begin, u_end = self.get_segment(doc_i)
            u_begin += doc_carry
            u_end += doc_carry+1
            u_indexes += list(range(u_begin, u_end))
            
        word_counts = np.sum(self.data.U_W_counts[u_indexes], axis=0)
        return word_counts
    
    def get_cluster_ll(self):
        return self.cluster_ll
    
    def set_cluster_ll(self, ll):
        self.cluster_ll = ll
        
    def get_doc_words(self, doc_i):
        wi_list = []
        u_begin, u_end = self.get_segment(doc_i)
        for u in range(u_begin, u_end+1):
            wi_list += self.data.d_u_wi_indexes[doc_i][u]
        return wi_list
    
    def get_segment(self, doc_i):
        '''
        Returns the first and last sentences (u_begin, u_end) of the doc_i
        document in this u_cluster 
        :param doc_i: index of the document
        '''
        seg_bound = self.doc_segs_dict[doc_i]
        u_begin = seg_bound[0]
        u_end = seg_bound[1]
        return u_begin, u_end
    
    def has_start_doc(self):
        for doc_i in self.doc_segs_dict:
            if self.doc_segs_dict[doc_i][0] == 0:
                return True
        return False
    