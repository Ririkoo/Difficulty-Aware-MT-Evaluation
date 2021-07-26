import numpy as np
from scipy.special import softmax

class BertScoreDifficulty:
    def __init__(self, encoded_ref):
        self.encoded_ref = encoded_ref
        self.token_weights = {}
        self.encoded_hypos = []
        self.sys_ref_status = []
        self.sys_hypo_status = []
        self.sys_ref_maxsim_loc = []
        self.sys_hypo_maxsim_loc = []

    def add_encoded_hypos(self, encoded_hypos):
        self.encoded_hypos.append(encoded_hypos)

    def add_ref_status(self, ref_status):
        self.sys_ref_status.append(ref_status)

    def add_hypo_status(self, hypo_status):
        self.sys_hypo_status.append(hypo_status)

    def add_ref_sloc(self, ref_maxsim_loc):
        self.sys_ref_maxsim_loc.append(ref_maxsim_loc)

    def add_hypo_sloc(self, hypo_maxsim_loc):
        self.sys_hypo_maxsim_loc.append(hypo_maxsim_loc)

    def cal_token_weights(self, weight):
        token_weights = {}
        for idx, token in enumerate(self.encoded_ref):
            if token not in token_weights.keys():
                token_weights[token] = weight[idx]
            else:
                token_weights[token] = (token_weights[token] + weight[idx]) / 2
        assert len(token_weights.keys()) == len(set(self.encoded_ref))
        self.token_weights = token_weights

    def normalized(self, weight_arr):
        max_val = np.max(weight_arr)
        if max_val == 0 or max_val < 1e-06:
            print('Non-valuable sample')
            return np.zeros(weight_arr.shape)
        else:
            # return softmax(weight_arr)
            return weight_arr/max_val

    def score_sent(self, softmax_norm=False, max_norm=False, range_one=False, ref_diff=False):
        sys_ref_status = np.array(self.sys_ref_status)
        weight = np.mean(sys_ref_status, axis=0)
        weight = 1.0 - weight.reshape(-1, 1)
        if softmax_norm:
            weight = softmax(weight)
        if max_norm:
            weight = self.normalized(weight)
        assert sys_ref_status.shape[1] == weight.shape[0]
        # Cal R
        sent_R = np.mean(sys_ref_status, axis=1)
        if np.sum(weight) == 0:
            weighted_sent_R = np.zeros([sys_ref_status.shape[0],1])
        else:
            if range_one:
                weighted_sent_R = np.dot(sys_ref_status, weight) / np.sum(weight) 
            else:
                weighted_sent_R = np.dot(sys_ref_status, weight) / weight.shape[0]
        self.cal_token_weights(weight)
        # Cal P
        sent_P = []
        weighted_sent_P = []
        for sys_no, sys_hypo in enumerate(self.sys_hypo_status):
            if len(sys_hypo) == 0:
                print('Empty Hypo, Set P=0')
                sent_P.append(0.0)
                weighted_sent_P.append(0.0)
                continue
            sent_P.append(sys_hypo.mean())
            encoded_sys_hypo = self.encoded_hypos[sys_no]
            sent_P_weight = np.ones(sys_hypo.shape[0])
            for loc, hypo_token in enumerate(encoded_sys_hypo):
                if hypo_token in self.token_weights.keys():
                    sent_P_weight[loc] = self.token_weights[hypo_token]
                elif ref_diff:
                    sent_P_weight[loc] = weight[self.sys_hypo_maxsim_loc[sys_no][loc]][0]
                
            if self.sys_hypo_maxsim_loc[sys_no].shape != sys_hypo.shape:
                ipdb.set_trace()
            assert len(encoded_sys_hypo) == sys_hypo.shape[0]
            assert sys_hypo.shape == sent_P_weight.shape
            if softmax_norm:
                sent_P_weight = softmax(sent_P_weight)
            if max_norm:
                sent_P_weight = self.normalized(sent_P_weight)
                
            if np.sum(sent_P_weight) == 0:
                weighted_P = 0
            else:
                if range_one:
                    weighted_P = np.sum(sys_hypo * sent_P_weight) / np.sum(sent_P_weight)
                else:
                    weighted_P = (sys_hypo * sent_P_weight).mean()
            weighted_sent_P.append(weighted_P)
        return np.array(sent_P), sent_R, np.array(
            weighted_sent_P), weighted_sent_R.reshape(-1)
