from config import ex
import numpy as np
from scipy.stats import truncnorm
import scipy.integrate
import scipy.stats
import random
import pandas as pd
import time
import warnings; warnings.simplefilter('ignore') #Ignores Warnings for nicer Plots. Disable for Debugging
import math
import data_utils
import os
import birkhoff
import relevance_network
from itertools import permutations
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plotting import *
import random
from Documents import Item, Movie
import copy
"""##User Affinity and Distribution"""

def assign_groups(items):
    n_groups = max([i.g for i in items])+1
    G = [ [] for i in range(n_groups)]
    for i, item in enumerate(items):
        G[item.g].append(i)
    return G

#Funktions for User score, position score, assigning groups and  User distributions
@ex.capture
def affinity_score(user, items, bernulli=True, DATA_SET=0):
    if DATA_SET == 1:
        if (type(items) == list):
            return np.asarray([user[0][x.id] for x in items])
        else:
            return user[0][items.id]
    elif DATA_SET == 0:
        #User normal distribution pdf without the normalization factor
        if(type(items) == list):
            item_affs = np.asarray([x.p for x in items])
            item_quality = np.asarray([x.q for x in items])
        else:
            item_affs = items.p
            item_quality = items.q

        #Calculating the Affnity Probability for each Item, based user polarity and user Openness
        aff_prob = np.exp(-(item_affs - user[0])**2 / (2*user[1]**2))*item_quality
        aff_prob1=aff_prob
        # Binarize The probability of Relevance to Actual relevance of an User
        aff_prob = np.random.rand(*np.shape(aff_prob)) < aff_prob
#         print(user,item_affs,item_quality,aff_prob1,aff_prob,"user,item_affs,item_quality,aff_prob1,aff_prob")
        return aff_prob
    
    
#    Calculate The position Bias for each of the N Positions #
def position_bias(n, model="PBM_log", ranked_relevances = None):
#     print(model=="PBM_log",model,"PBM_log")
    if(model=="PBM_log"):#Position Based Model with inverse log(rank)
        pos = 1/(np.log2(2+np.arange(n))) 
        pos /= np.max(pos)
    elif(model=="PBM_cutoff"):#Position Based Model with inverse log(rank)
        pos=np.zeros(n)
        top_n = 1/(np.log2(2+np.arange(10)))
        top_n /= np.max(top_n)
        pos[:10]=top_n
    elif(model=="PBM_inv"):#Position Based Model with inverse rank
        scale = 1
        pos = (1/(1+ np.arange(n)))**scale
    elif(model=="Cascade" or model=="DCM"):
        assert(ranked_relevances is not None)
        if(model=="Cascade"): #Cascade Model
            gamma_click = 0
            gamma_no = 1
        else: #Dependent Click Model
            gamma_click = 0.5
            gamma_no = 0.9
        if(np.max(ranked_relevances) >1):
            ranked_relevances = ranked_relevances / np.max(ranked_relevances)
        pos = np.ones(n)
        for i in range(1, len(pos)):
            pos[i] = pos[i-1]* (gamma_click * ranked_relevances[i-1]+ gamma_no* (1-ranked_relevances[i-1]))
    elif model == "PBM_TEST":
        pos = np.ones(n)
    else:
        print("Could not find", model)
    return pos


###### Calculate NDCG Score

def get_ndcg_score(ranking, true_relevances, click_model = "PBM_log"):
    dcg = np.sum(true_relevances[ranking] / np.log2(2+np.arange(len(ranking))))
    idcg = np.sum(np.sort(true_relevances)[::-1] / np.log2(2+np.arange(len(ranking))))
    if dcg is None or idcg is None or dcg/idcg is None:
        print("Some kind of None appeard with",dcg, idcg, dcg/idcg)
    if(idcg ==0):
        return 1
    return dcg / idcg

def get_ndcg_score_top_k(ranking, true_relevances, click_model = "PBM_log",top_k_list=[1,3]):
    ndcg_top_k=[]
    for i in top_k_list:
#         print(true_relevances[ranking])
        dcg = np.sum(true_relevances[ranking][:i] / np.log2(2+np.arange(len(ranking)))[:i])
        idcg = np.sum(np.sort(true_relevances)[::-1][:i] / np.log2(2+np.arange(len(ranking)))[:i])
        if dcg is None or idcg is None or dcg/idcg is None:
            print("Some kind of None appeard with",dcg, idcg, dcg/idcg)
        if(idcg ==0):
            ndcg_top_k.append(1)
        else:
            ndcg_top_k.append(dcg / idcg)
    ndcg_top_k=np.array(ndcg_top_k)
    return ndcg_top_k

@ex.capture
def get_numerical_relevances(items, DATA_SET, MOVIE_RATING_FILE=""):
    if DATA_SET == 0:
        users = [data_utils.sample_user_base(distribution="bimodal") for i in range(50000)]
        aff = [affinity_score(u, items, DATA_SET=DATA_SET) for u in users]
        return np.mean(np.asarray(aff), axis=0)
    elif DATA_SET == 1:
        ranking, _, _ = data_utils.load_movie_data_saved(MOVIE_RATING_FILE)
        return np.mean(ranking, axis=0)  # Mean over all users

#Function to obtain a new users, Depending on the Dataset

class Usersampler:
    @ex.capture
    def __init__(self, DATA_SET, BI_LEFT, MOVIE_RATING_FILE):
        self.data_set = DATA_SET
        if DATA_SET == 1:
            self.sample_user_generator = data_utils.sample_user_movie(MOVIE_RATING_FILE)
        if DATA_SET == 0:
            self.BI_LEFT = BI_LEFT

    def get_user(self):
        if self.data_set == 0:
            return data_utils.sample_user_base(distribution="bimodal", BI_LEFT=self.BI_LEFT)
        elif self.data_set == 1:
            return next(self.sample_user_generator)


def top_down(weighted_popularity,G,rank_group_split,personal_relevance=np.array([-100]),add_args=None):
    average_rele=[]
#     print("using top down")
#     print(add_args,"in top down","*"*50)
    pos_bias=position_bias(weighted_popularity.shape[0])
    if np.sum(personal_relevance)==-100:
#         print("no personal relevance using average relevance")
        personal_relevance=weighted_popularity
#     print(weighted_popularity,personal_relevance,"weighted_popularity,personal_relevance")
    for i in G:
        average_rele.append(np.mean(weighted_popularity[i]))
    average_rele=np.array(average_rele)
#     print(average_rele,"average_rele","weighted_popularity",weighted_popularity)
    rank_group_split_copy = copy.deepcopy(rank_group_split)
    G_size=np.array([len(G_i) for G_i in G])
    sorted_G=[]
    def func(G):
        return personal_relevance[G]
    for i in G:
        sorted_G.append(sorted(i, key = func,reverse=True))
#     rank_group_split
    def cum_id(rank_group_split_copy,k):
        product=rank_group_split_copy*pos_bias[np.newaxis,:]
        rank_group_split_cum_exposure=np.cumsum(product,axis=1)[:,k]/average_rele/G_size
#         print(rank_group_split_cum_exposure,"rank_group_split_cum_exposure","*"*20)
#         print(average_rele,"average_rele","*"*20)
#         print(np.cumsum(product,axis=1)[:,k],"np.cumsum(product,axis=1)[:,k]","*"*20)
        sorted_group_id=rank_group_split_cum_exposure.argsort(0)
        return sorted_group_id
#     rank_group_split_cum=np.cumsum(rank_group_split,axis=1)    
#     rank_group_split_cum_exposure=rank_group_split_cum/average_rele[:,np.newaxis]    
#     sorted_group_id=rank_group_split_cum_exposure.argsort(0)
    rangking=[]
    lambda_random_whole=2
    
    lambda_random_whole_part=0.5
    if "lambda_fair"in add_args.keys():
        lambda_random_whole_part=add_args["lambda_fair"]
#     print(lambda_random_whole_part,"lambda_random_whole_part","*"*30)
    ranking_relevance=np.argsort(personal_relevance)[::-1]
#     print(weighted_popularity,"weighted_popularity\n","*"*20)
#     print(personal_relevance,"personal_relevance\n","*"*20)
#     print(average_rele,"average_rele\n","*"*20)
#     print(sorted_G,"sorted_G\n","*"*20)
#     print(G,"G\n","*"*20)
#     print(rank_group_split,"rank_group_splitn","*"*20)
    sorted_group_id_list=[]
    if random.random()>lambda_random_whole:
        return ranking_relevance
    else:
        for i in range(weighted_popularity.shape[0]):
            if random.random()>lambda_random_whole_part:
                G_id=get_maximu_G_id(sorted_G,personal_relevance)
                rangking.append(sorted_G[G_id].pop(0))
                rank_group_split_copy[G_id,i]+=1
                
            else:
                sorted_group_id=cum_id(rank_group_split_copy,i)
                sorted_group_id_list.append(sorted_group_id)
    #             print(sorted_group_id,"sorted_group_id\n","*"*20)
                for j in range(len(G)):
                    if sorted_G[sorted_group_id[j]]:
                        rangking.append(sorted_G[sorted_group_id[j]].pop(0))
                        rank_group_split_copy[sorted_group_id[j],i]+=1
                        break
        ranking=np.array(rangking).astype(int)
#         print(ranking,"ranking\n","*"*20)
#         sorted_group_id_list=np.array(sorted_group_id_list)
#         print(rank_group_split_copy,rank_group_split,sorted_group_id_list,\
#               "rank_group_split_copy,rank_group_split,sorted_group_id")
        return ranking

def get_maximu_G_id(sorted_G,personal_relevance):
    max_G_ind=-1000
    max_i_value=-1000
    for ind,i in enumerate(sorted_G):
        if len(i)>0:
#             print(i)
            if max_i_value<personal_relevance[i[0]]:
                max_i_value=personal_relevance[i[0]] 
                max_G_ind=ind
    return max_G_ind        
        
def max_min_loss(a):
    return np.max(a)-np.min(a)
def ranking_2_average_exp(G_id_ranking_list,G):
    exposure_aver=[]
    G_id_ranking_list=np.array(G_id_ranking_list)
    for G_i in range(len(G)):
        ind_Gi=G_id_ranking_list==G_i
        exposure_cur=np.sum(ind_Gi*1*position_bias(len(G_id_ranking_list)))
        exposure_aver.append(exposure_cur/(np.sum(ind_Gi)+1e-10))
    exposure_aver=np.array(exposure_aver)
    
    return exposure_aver
def Greedy_selection(weighted_popularity,G,rank_group_split,\
                     personal_relevance=np.array([-100]),add_args=None):
    average_rele=[]
#     print("using greedy")
#     print(add_args,"in top down","*"*50)
    pos_bias=position_bias(weighted_popularity.shape[0])
    if np.sum(personal_relevance)==-100:
#         print("no personal relevance using average relevance")
        personal_relevance=weighted_popularity
#     print(weighted_popularity,personal_relevance,"weighted_popularity,personal_relevance")
    for i in G:
        average_rele.append(np.mean(weighted_popularity[i]))
    average_rele=np.array(average_rele)
    rank_group_split_copy = copy.deepcopy(rank_group_split)
    accum_aver_exp=copy.deepcopy(add_args["accum_aver_exp"])
    sorted_G=[]
    def func(G):
        return personal_relevance[G]
    for i in G:
        sorted_G.append(sorted(i, key = func,reverse=True))
#     rank_group_split
    def cum_id(rank_group_split_copy,k):
        product=rank_group_split_copy*pos_bias[np.newaxis,:]
        rank_group_split_cum_exposure=np.cumsum(product,axis=1)[:,k]/average_rele
        sorted_group_id=rank_group_split_cum_exposure.argsort(0)
        return sorted_group_id
#     rank_group_split_cum=np.cumsum(rank_group_split,axis=1)    
#     rank_group_split_cum_exposure=rank_group_split_cum/average_rele[:,np.newaxis]    
#     sorted_group_id=rank_group_split_cum_exposure.argsort(0)
    rangking=[]
    lambda_random_whole=2
    lambda_random_whole_part=0.5
    if "lambda_fair"in add_args.keys():
        lambda_random_whole_part=add_args["lambda_fair"]
#     print(lambda_random_whole_part,"lambda_random_whole_part","*"*30)
    ranking_relevance=np.argsort(personal_relevance)[::-1]
#     print(weighted_popularity,"weighted_popularity\n","*"*20)
#     print(personal_relevance,"personal_relevance\n","*"*20)
#     print(average_rele,"average_rele\n","*"*20)
#     print(sorted_G,"sorted_G\n","*"*20)
#     print(G,"G\n","*"*20)
#     print(rank_group_split,"rank_group_splitn","*"*20)
    sorted_group_id_list=[]
    G_id_ranking_list=[]
    if random.random()>lambda_random_whole:
        return ranking_relevance
    else:
        for i in range(weighted_popularity.shape[0]):
            if random.random()>lambda_random_whole_part:
                G_id=get_maximu_G_id(sorted_G,personal_relevance)
                G_id_ranking_list.append(G_id)
                rangking.append(sorted_G[G_id].pop(0))
                rank_group_split_copy[G_id,i]+=1
                
            else:
                sorted_group_id=cum_id(rank_group_split_copy,i)
                sorted_group_id_list.append(sorted_group_id)
    #             print(sorted_group_id,"sorted_group_id\n","*"*20)
                loss=math.inf
                selected_G_id=0
                for j in range(len(G)):
                    if sorted_G[j]:
                        aver_exp=ranking_2_average_exp(rangking+[j],G)
                        loss_cur=max_min_loss((aver_exp+accum_aver_exp[:,i])/average_rele)
                        if loss_cur<=loss:
                            loss=loss_cur
                            selected_G_id=j
                
                rangking.append(sorted_G[selected_G_id].pop(0))
#                         rank_group_split_copy[sorted_group_id[j],i]+=1
                        
        ranking=np.array(rangking).astype(int)
#         print(ranking,"ranking\n","*"*20)
#         sorted_group_id_list=np.array(sorted_group_id_list)
#         print(rank_group_split_copy,rank_group_split,sorted_group_id_list,\
#               "rank_group_split_copy,rank_group_split,sorted_group_id")
        return ranking      

def get_ranking(user, popularity, items, weighted_popularity=None, G=None, ranking_method="Naive", click_model="PBM_log",
                cum_exposure=None, decomp=None, new_fair_rank=False, nn=None, integral_fairness=None,relevance_ground=None,rank_group_split=None,rank_group_split_click=None,add_args=None):
    """
    Get the Ranking and position Bias
    For the Linear Program, we also return the current ranking Decomposition (decomp)
    For Fairness Controlling programs, we also return the Fairness Error (fairess_error)
    """
    n = len(popularity)
    click_prob = np.zeros(n)
    fairness_error = None

    # Ranking of the entries
    if (ranking_method == "Naive"):
        ranking = pop_rank(popularity)
    elif (ranking_method == "IPS_I_top_down"):
        ranking = top_down(weighted_popularity,G,rank_group_split_click,add_args=add_args)
    elif (ranking_method == "IPS_E_top_down"):
        ranking = top_down(weighted_popularity,G,rank_group_split,add_args=add_args)        
    elif (ranking_method == "IPS_E_greedy"):
        ranking = Greedy_selection(weighted_popularity,G,rank_group_split,add_args=add_args)           
    elif (ranking_method == "IPS"):
        assert (weighted_popularity is not None)
        ranking = IPS_rank(weighted_popularity)
#     if (ranking_method == "Naive"):
#         ranking = pop_rank(relevance_ground)
#     elif (ranking_method == "IPS"):
#         assert (weighted_popularity is not None)
#         ranking = IPS_rank(relevance_ground)
        
    elif ("IPS-LP" in ranking_method and "Pers" not in ranking_method):
        # Try Linear Programm for fair ranking, when this fails, use last ranking
        if new_fair_rank or decomp is None:
            if (ranking_method == "Fair-E-IPS-LP"):
                group_fairness = get_unfairness(cum_exposure, weighted_popularity, G, error=False)
                decomp = fair_rank(items, weighted_popularity, debug=False,
                                   group_click_rel=group_fairness, impact=False)
            elif (ranking_method == "Fair-I-IPS-LP"):
                group_fairness = get_unfairness(popularity, weighted_popularity, G, error=False)
                decomp = fair_rank(items, weighted_popularity, debug=False,
                                   group_click_rel=group_fairness, impact=True)
            else:
                raise Exception("Unknown Fair method specified")
        if decomp is not None:
            p_birkhoff = np.asarray([np.max([0, x[0]]) for x in decomp])
            p_birkhoff /= np.sum(p_birkhoff)
            sampled_r = np.random.choice(range(len(decomp)), 1, p=p_birkhoff)[0]
            ranking = np.argmax(decomp[sampled_r][1], axis=0)
        else:
            ranking = IPS_rank(weighted_popularity)
    elif (ranking_method == "Fair-I-IPS"):
        fairness_error = get_unfairness(popularity, weighted_popularity, G, error=True)
        ranking = controller_rank(weighted_popularity, fairness_error)

#     elif (ranking_method == "Fair-I-IPS"):
#         fairness_error = get_unfairness(popularity, relevance_ground, G, error=True)
#         ranking = controller_rank(relevance_ground, fairness_error)
    elif (ranking_method == "Fair-E-IPS"):
        fairness_error = get_unfairness(cum_exposure, weighted_popularity, G, error=True)
        ranking = controller_rank(weighted_popularity, fairness_error)
    elif ("Pers" in ranking_method):
        
        KP=0.01
#         print(ranking_method,"ranking_method")
        if nn is None:
            ranking = IPS_rank(weighted_popularity)
        elif "Fair-E-Pers" == ranking_method:
            fairness_error = get_unfairness(cum_exposure, weighted_popularity, G, error=True)
            ranking = neural_rank(nn, items, user, e_p=fairness_error,KP= KP)
#             ranking = neural_rank(nn, items, user,KP= KP)
        elif "Fair-I-Pers" == ranking_method:
            fairness_error = get_unfairness(popularity, weighted_popularity, G, error=True)
            ranking = neural_rank(nn, items, user, e_p=fairness_error,KP= KP)
        elif "Fair-E_top_down-Pers" == ranking_method:
            ranking = neural_rank_top_down(nn, items, user,weighted_popularity=weighted_popularity,\
                                           G=G,rank_group_split=rank_group_split,add_args=add_args) 
        elif "Fair-E_greedy-Pers" == ranking_method:
            ranking = neural_rank_Greedy(nn, items, user,weighted_popularity=weighted_popularity,\
                                           G=G,rank_group_split=rank_group_split,add_args=add_args) 
        elif "Fair-I_top_down-Pers" == ranking_method:
            ranking = neural_rank_top_down(nn, items, user, weighted_popularity=weighted_popularity,\
                                           G=G,rank_group_split=rank_group_split_click,add_args=add_args)  
        elif (ranking_method == "Fair-E-IPS_Pers-LP"):
                DATA_SET=1
                if DATA_SET == 1 :
                    x_test = np.asarray(user[1])
                elif DATA_SET == 0:
                    x_test = np.asarray(user)
                relevances = nn.predict(x_test)
                group_fairness = get_unfairness(cum_exposure, weighted_popularity, G, error=False)
                decomp = fair_rank_Pers(items, weighted_popularity,relevances,debug=False,
                                   group_click_rel=group_fairness, impact=False)
                if decomp is not None:
                    print("fair rank per sucessful for user")
                    p_birkhoff = np.asarray([np.max([0, x[0]]) for x in decomp])
                    p_birkhoff /= np.sum(p_birkhoff)
                    sampled_r = np.random.choice(range(len(decomp)), 1, p=p_birkhoff)[0]
                    ranking = np.argmax(decomp[sampled_r][1], axis=0)
                else:
                    print("fair rank per fail for user")
                    ranking = IPS_rank(weighted_popularity)
        else:
            ranking = neural_rank(nn, items, user,KP= KP)
    elif (ranking_method == "Random"):
        ranking = random_rank(weighted_popularity)
    else:
        print("could not find a ranking method called: " + ranking_method)
        raise Exception("No Method specified")
## Above part is to generate the ranking, and below is used to collect feedback.
    # create prob of click based on position
    pos = position_bias(n, click_model, weighted_popularity[ranking])

    # reorder position probabilities to match popularity order
    pos_prob = np.zeros(n)
    pos_prob[ranking] = pos
    return pos_prob, ranking, decomp, fairness_error


def get_unfairness(clicks, rel, G, error=False):
    """
    Get the Unfairess
    Input Clicks (Cum_Exposure for Exposure Unfairness, Clicks for Impact Unfairness)
    If Error, we return the difference to the best treated group,
    Otherwise just return the Exposure/Impact per Relevance
    """
    n = len(clicks)
    group_clicks = [sum(clicks[G[i]]) for i in range(len(G))]
    group_rel = [max(0.0001, sum(rel[G[i]])) for i in range(len(G))]
    group_fairness = [group_clicks[i] / group_rel[i] for i in range(len(G))]
    if (error):
        best = np.max(group_fairness)
        fairness_error = np.zeros(n)
        for i in range(len(G)):
            fairness_error[G[i]] = best - group_fairness[i]
        return fairness_error
    else:
        return group_fairness


# simulation function returns number of iterations until convergence
@ex.capture
def simulate(popularity, items, ranking_method="Naive", click_model="PBM_log", iterations=2000,
             numerical_relevance=None, head_start=-1, DATA_SET=0, HIDDEN_UNITS=64, PLOT_PREFIX="", user_generator=None,add_args=None):
    #global sample_user
    """
    :param popularity: Initial Popularity
    :param items:  Items/Documents
    :param ranking_method: Method to Use: eg. Naiva, IPS, Pers, Fair-I
    :param click_model: Clickmodel  (PBM_log)
    :param iterations: Iterations/User to sample
    :param numerical_relevance: Use numerical relevance or sampled
    :return count, hist, pophist, ranking, users, ideal_ranking, mean_relevances, w_pophist, nn_errors, mean_exposure, fairness_hist, p_pophist:
    count: Iterations run
    hist: Ranking History
    pophist: Click_History
    ranking: Final ranking
    users: Users sampled
    ideal_ranking: Optimal Ranking
    mean_relevances: Mean Relevance per Item
    w_pophist: Weighted IPS Rating
    nn_errors: Error of Neural Network
    mean_exposure: Mean Exposure per Item
    fairness_hist: Propensities, clicks, estimated_relevance, true_rel per Group and  NDCG
    p_pophist: Personalized Relevance history
    """
    #Initialize Variables
    G = assign_groups(items)
    group_dict={}
    for ind,i in enumerate(G):
        for j in i:
            group_dict[j]=ind
    def dict_table(a):
        return group_dict[a]
    dict_table_v_func = np.vectorize(dict_table)
    weighted_popularity = np.asarray(popularity, dtype=np.float32)
    popularity = np.asarray(popularity)
    pophist = np.zeros((iterations, len(items)))
    w_pophist = np.zeros((iterations, len(items)))
    rank_group_split=np.zeros((len(G),len(items)))
    if "Pers" in ranking_method:
        p_pophist = np.zeros((iterations, len(items)))
    else:
        p_pophist = None
    users = []
#     print(click_model,"click_model")
    aff_scores = np.zeros((iterations, len(items)))
    relevances = np.zeros(len(items))
    cum_exposure = np.zeros(len(items))
    hist = np.zeros((iterations, len(popularity)))
    propensity_history = np.zeros((iterations, len(popularity)))
    decomp = None
    group_prop = np.zeros((iterations, len(G)))
    group_clicks = np.zeros((iterations, len(G)))
    group_rel = np.zeros((iterations, len(G)))
    true_group_rel = np.zeros((iterations, len(G)))
    cum_fairness_error = np.zeros(len(items))
    rank_hist = np.zeros((iterations, len(items)))
    rank_group_id_sorted = np.zeros((iterations, len(items)))
    rank_group_split_click=np.zeros((len(G),len(items)))
    accum_aver_exp=np.zeros((len(G),len(items)))
    if "top_k_list" in add_args.keys():
        print(len(add_args["top_k_list"]))
        ndcg_top_k=np.zeros((iterations,len(add_args["top_k_list"])))
    NDCG = np.zeros(iterations)
    add_args=copy.deepcopy(add_args)
    if (numerical_relevance is None):
        numerical_relevance = get_numerical_relevances(items)
#     print(popularity,"popularity")
    # counters
    count = 0
    nn_errors = np.zeros(iterations)
    nn = None
    if user_generator is None:
        user_generator = Usersampler()
    pos_bias=position_bias(len(items))
    for i in range(iterations):
        if add_args:
            add_args["accum_aver_exp"]=accum_aver_exp
        count += 1
        #For the Headstart Experiment, we first choose Right then Left Leaning Users
        if (i <= head_start * 2):
            if i == head_start * 2:
                user_generator = Usersampler(BI_LEFT=0.5)
            elif i < head_start:
                user_generator = Usersampler(BI_LEFT=0)
            else:
                user_generator = Usersampler(BI_LEFT=1)

        # choose user
        user = user_generator.get_user()
        users.append(user)
#         print(iterations,"iterations")
        aff_probs = affinity_score(user, items, DATA_SET=DATA_SET)
        relevances += aff_probs

        # clicking probabilities
        propensities, ranking, decomp, fairness_error = get_ranking(user, popularity, items, weighted_popularity / count, G,
                                                                    ranking_method, click_model, cum_exposure, decomp,
                                                                    count % 100 == 9, nn=nn,
                                                                    integral_fairness=cum_fairness_error / count,relevance_ground=numerical_relevance,rank_group_split=rank_group_split,rank_group_split_click=rank_group_split_click,add_args=add_args)
#         print(ranking,"iterations ",flush=True)
        
        # update popularity
        popularity, weighted_popularity,clicks = simulate_click(aff_probs, propensities, popularity, weighted_popularity,
                                                         ranking, click_model)

        # Save History
        rank_hist[i]=ranking
        aff_scores[i] = aff_probs
        hist[i, :] = ranking
        cum_exposure += propensities
        pophist[i, :] = popularity
        w_pophist[i, :] = weighted_popularity
        propensity_history[i, :]=propensities
        rank_group_id_sorted[i,:]=dict_table_v_func(ranking)
        for G_i in range(len(G)):
            ind_Gi=rank_group_id_sorted[i]==G_i
            exposure_cur=np.cumsum(ind_Gi*1*pos_bias)
            exposure_aver=exposure_cur/(np.cumsum(ind_Gi)+1e-10)
    #         print(ind_Gi,exposure_cur,np.cumsum(ind_Gi)+1e-10)
    #         print(exposure_aver)
            accum_aver_exp[G_i]+=exposure_aver          
            rank_group_split[G_i,rank_group_id_sorted[i]==G_i]+=1
            clicks_ranked=clicks[ranking]
            ind_clicked=np.all([rank_group_id_sorted[i]==G_i,clicks_ranked==1],0)
            rank_group_split_click[G_i,ind_clicked]+=1
        # update neural network
        if "Pers" in ranking_method:
            if (i == 99):  # Initialize Neural Network
                if DATA_SET == 0:
                    train_x = np.asarray(users)
                elif DATA_SET == 1:
                    train_x = np.asarray([u[1] for u in users])
                if not "Skyline" in ranking_method:
                    nn = relevance_network.relevance_estimating_network(np.shape(train_x)[1], output_dim=len(items),
                                                                        hidden_units=HIDDEN_UNITS,
                                                                        news=True,
                                                                        logdir=PLOT_PREFIX)
                    train_y = w_pophist[:i + 1] - np.concatenate((np.zeros((1, len(items))), w_pophist[:i]))
                else:
                    # Supervised Baseline
                    nn = relevance_network.relevance_estimating_network(np.shape(train_x)[1], output_dim=len(items),
                                                                        hidden_units=HIDDEN_UNITS,
                                                                        news=True,
                                                                        supervised=True, logdir=PLOT_PREFIX)
                    train_y = aff_scores[:i + 1]  ## Ground truth relevance score.
                nn.train(train_x, train_y, epochs=2000, trial=i)
            elif (i > 99 and i % 10 == 9):
                if "Skyline" in ranking_method:
                    train_y = aff_scores[:i + 1]
                else:
                    train_y = np.concatenate((train_y, w_pophist[i - 9:i + 1] - w_pophist[i - 10:i]))
                if DATA_SET == 1:
                    train_x = np.concatenate((train_x, np.asarray([u[1] for u in users[-10:]])))
                else:
                    train_x = np.concatenate((train_x, np.asarray([u for u in users[-10:]])))

                nn.train(train_x, train_y, epochs=10, trial=i)

            if DATA_SET and i >= 99:
                predicted_relevances = nn.predict(user[1])
            elif i >= 99:
                predicted_relevances = nn.predict(user)
            if i >= 99:
                nn_errors[i] = np.mean((predicted_relevances - aff_probs) ** 2)
                p_pophist[i, :] = predicted_relevances
            else:
                p_pophist[i, :] = weighted_popularity


        # Save statistics
        if (fairness_error is not None):
            cum_fairness_error += fairness_error

        if DATA_SET:
            NDCG[i] = get_ndcg_score(ranking, user[0])
            if "top_k_list" in add_args.keys():
                ndcg_top_k[i]=get_ndcg_score_top_k(ranking, user[0],top_k_list=add_args["top_k_list"])
        else:
            NDCG[i] = get_ndcg_score(ranking, aff_probs)
            if "top_k_list" in add_args.keys():
                ndcg_top_k[i]=get_ndcg_score_top_k(ranking, aff_probs,top_k_list=add_args["top_k_list"])
        group_prop[i, :] = [np.sum(cum_exposure[G[i]]) for i in range(len(G))]
        group_clicks[i, :] = [np.sum(popularity[G[i]]) for i in range(len(G))]
        if ("Pers" in ranking_method):
            group_rel[i, :] = [np.sum(p_pophist[i, G[g]]) for g in range(len(G))]
        elif ("Naive" in ranking_method):
            group_rel[i, :] = [np.sum(pophist[i, G[g]]) for g in range(len(G))]
        else:
            group_rel[i, :] = [np.sum(weighted_popularity[G[g]]) for g in range(len(G))]

        true_group_rel[i, :] = [np.sum(numerical_relevance[G[g]]) * count for g in range(len(G))]


    ideal_vals, ideal_ranking = ideal_rank(users, items, DATA_SET = DATA_SET)

    mean_relevances = relevances / count
    mean_exposure = cum_exposure / count

    fairness_hist = {"prop": group_prop, "clicks": group_clicks, "rel": group_rel, "true_rel": true_group_rel,
                     "NDCG": NDCG,"aff_scores":aff_scores,"pophist":pophist,\
                     "hist":hist,"w_pophist":w_pophist,"propensity_history":propensity_history,\
                     "G":G,"rank_hist":rank_hist,"numerical_relevance":numerical_relevance,\
                     "rank_group_id_sorted":rank_group_id_sorted,"rank_group_split":rank_group_split,\
                    "rank_group_split_click":rank_group_split_click}
    if "top_k_list" in add_args.keys():
        for ind,top_k in enumerate(add_args["top_k_list"]):
            fairness_hist["NDCG_"+str(top_k)]=ndcg_top_k[:,ind]
    return count, hist, pophist, ranking, users, ideal_ranking, mean_relevances, w_pophist, nn_errors, mean_exposure, fairness_hist, p_pophist


def simulate_click(aff_probs, propensities, popularity, weighted_popularity, ranking, click_model):
    if "PBM" in click_model:
        rand_var = np.random.rand(len(aff_probs))
        rand_prop = np.random.rand(len(propensities))
        viewed = rand_prop < propensities
        clicks = np.logical_and(rand_var < aff_probs, viewed)
        popularity += clicks
        weighted_popularity += clicks / (propensities+1e-5)  ##
        
    elif click_model == "Cascade" or click_model == "DCM":
        c_stop = 1
        if click_model == "Cascade":
            gamma_click = 0
            gamma_no = 1
        else:
            gamma_click = 0.5
            gamma_no = 0.98
        for i, r in enumerate(ranking):
            if random.random() < aff_probs[r]:
                popularity[r] += 1
                weighted_popularity[r] += 1. / c_stop
                c_stop *= gamma_click
                if random.random() > gamma_click:
                    break
            else:
                if random.random() > gamma_no:
                    break
                c_stop *= gamma_no
    else:
        raise Exception("Could not find the clickmodel")
    return popularity, weighted_popularity,clicks


"""##Ranking Functions"""
#Ranking Functions:
#Popularity Ranking
def pop_rank(popularity):
    return np.argsort(popularity)[::-1]

#Inverse Propensity Ranking
def IPS_rank(weighted_popularity):
    return np.argsort(weighted_popularity)[::-1]

#Random Ranking
def random_rank(weighted_popularity):
    ranking = np.arange(len(weighted_popularity))
    np.random.shuffle(ranking)
    return ranking

#Rank using a simple P Controller
@ex.capture
def controller_rank(weighted_popularity, e_p, KP= 0.01):
    return np.argsort(weighted_popularity + KP * e_p )[::-1]

#Ranking with neural network relevances
@ex.capture
def neural_rank(nn, items, user, DATA_SET = 1, e_p = 0, KP= 0.01 ):
#     print(KP)
    if DATA_SET == 1 :
        x_test = np.asarray(user[1])
    elif DATA_SET == 0:
        x_test = np.asarray(user)
    relevances = nn.predict(x_test)
    return np.argsort(relevances+ KP * e_p)[::-1]

#Ranking with neural network relevances
@ex.capture
def neural_rank_top_down(nn, items, user, DATA_SET = 1, weighted_popularity=None,G = 0, rank_group_split= 0.01,add_args=None ):
#     print(KP)
    if DATA_SET == 1 :
        x_test = np.asarray(user[1])
    elif DATA_SET == 0:
        x_test = np.asarray(user)
    relevances = nn.predict(x_test)
    
    return top_down(weighted_popularity,G,rank_group_split,relevances,add_args=add_args)
@ex.capture
def neural_rank_Greedy(nn, items, user, DATA_SET = 1, weighted_popularity=None,G = 0, rank_group_split= 0.01,add_args=None ):
#     print(KP)
    if DATA_SET == 1 :
        x_test = np.asarray(user[1])
    elif DATA_SET == 0:
        x_test = np.asarray(user)
    relevances = nn.predict(x_test)
    
    return Greedy_selection(weighted_popularity,G,rank_group_split,relevances,add_args=add_args)

#Fair Ranking
@ex.capture
def fair_rank(items, popularity,ind_fair=False, group_fair=True, debug=False, w_fair = 1, group_click_rel = None, impact=True, LP_COMPENSATE_W=10):
    n = len(items)
    pos_bias = position_bias(n)
    G = assign_groups(items)
    n_g, n_i = 0, 0
    if(group_fair):
        n_g += (len(G)-1)*len(G)
    if(ind_fair):
        n_i += n * (n-1)

    n_c = n**2 + n_g + n_i


    c = np.ones(n_c)
    c[:n**2] *= -1
    c[n**2:] *= w_fair
    A_eq = []
    #For each Row
    for i in range(n):
        A_temp = np.zeros(n_c)
        A_temp[i*n:(i+1)*n] = 1
        assert(sum(A_temp)==n)
        A_eq.append(A_temp)
        c[i*n:(i+1)*n] *= popularity[i]

    #For each coloumn
    for i in range(n):
        A_temp = np.zeros(n_c)
        A_temp[i:n**2:n] = 1
        assert(sum(A_temp)==n)
        A_eq.append(A_temp)
        #Optimization
        c[i:n**2:n] *= pos_bias[i]
    b_eq = np.ones(n*2)
    A_eq = np.asarray(A_eq)
    bounds = [(0,1) for _ in range(n**2)] + [(0,None) for _ in range(n_g+n_i)]


    A_ub = []
    b_ub = np.zeros(n_g+n_i)
    if(group_fair):
        U = []
        for group in G:
            #Avoid devision by zero
            u = np.max([sum(np.asarray(popularity)[group]), 0.01])
            U.append(u)
        comparisons = list(permutations(np.arange(len(G)),2))
        j = 0
        for a,b in comparisons:
            f = np.zeros(n_c)
            if len(G[a]) > 0 and len(G[b])>0:
                for i in range(n):
                    if impact:
                        tmp1 = popularity[i] / U[a] if i in G[a] else 0
                        tmp2 = popularity[i] / U[b] if i in G[b] else 0
                    else:
                        tmp1 = 1. / U[a] if i in G[a] else 0
                        tmp2 = 1. / U[b] if i in G[b] else 0
                    f[i*n:(i+1)*n] =  (tmp1 - tmp2) # * popularity[i] for equal impact instead of equal Exposure
                for i in range(n):
                    f[i:n**2:n] *= pos_bias[i]
                f[n**2+j] = -1
                if group_click_rel is not None:
                    b_ub[j] = LP_COMPENSATE_W * (group_click_rel[b] - group_click_rel[a])
            j += 1
            A_ub.append(f)

    if(ind_fair):
        comparisons = list(permutations(np.arange(len(popularity)),2))
        j = 0
        for a,b in comparisons:
            f = np.zeros(n_c)
            if(popularity[a] >= popularity[b]):
                tmp1 = 1. / np.max([0.01,popularity[a]])
                tmp2 = 1. / np.max([0.01,popularity[b]])
                f[a*n:(a+1)*n] = tmp1
                f[a*n:(a+1)*n] *= pos_bias
                f[b*n:(b+1)*n] = -1 *  tmp2
                f[b*n:(b+1)*n] *= pos_bias

                f[n**2+n_g+j] = -1
            j += 1
            A_ub.append(f)

    res = scipy.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, options=dict(bland =True, tol=1e-12), method = "interior-point")
    probabilistic_ranking = np.reshape(res.x[:n**2],(n,n))


    if(debug):
        print("Shape of the constrains", np.shape(A_eq), "with {} items and {} groups".format(n, len(G)))
        print("Fairness constraint:", np.round(np.dot(A_eq,res.x),4))
        #print("Constructed probabilistic_ranking with score {}: \n".format(res.fun), np.round(probabilistic_ranking,2))
        print("Col sum: ", np.sum(probabilistic_ranking,axis=0))
        print("Row sum: ", np.sum(probabilistic_ranking,axis=1))
        #plt.matshow(A_eq)
        #plt.colorbar()
        #plt.plot()
        plt.matshow(probabilistic_ranking)
        plt.colorbar()
        plt.plot()

    #Sample from probabilistic ranking using Birkhoff-von-Neumann decomposition
    try:
        decomp = birkhoff.birkhoff_von_neumann_decomposition(probabilistic_ranking)
    except:
        decomp = birkhoff.approx_birkhoff_von_neumann_decomposition(probabilistic_ranking)

        if debug:
            print("Could get a approx decomposition with {}% accuracy".format(100*sum([x[0] for x in decomp])) )
            #print(probabilistic_ranking)

    return decomp
#Fair Ranking
@ex.capture
def fair_rank_Pers(items, popularity,popularity_per,ind_fair=False, group_fair=True, debug=False, w_fair = 1, group_click_rel = None, impact=True, LP_COMPENSATE_W=10):
    print("using fair rank per")
    n = len(items)
    pos_bias = position_bias(n)
    G = assign_groups(items)
    n_g, n_i = 0, 0
    if(group_fair):
        n_g += (len(G)-1)*len(G)
    if(ind_fair):
        n_i += n * (n-1)

    n_c = n**2 + n_g + n_i


    c = np.ones(n_c)
    c[:n**2] *= -1
    c[n**2:] *= w_fair
    A_eq = []
    #For each Row
    for i in range(n):
        A_temp = np.zeros(n_c)
        A_temp[i*n:(i+1)*n] = 1
        assert(sum(A_temp)==n)
        A_eq.append(A_temp)
        c[i*n:(i+1)*n] *= popularity_per[i]

    #For each coloumn
    for i in range(n):
        A_temp = np.zeros(n_c)
        A_temp[i:n**2:n] = 1
        assert(sum(A_temp)==n)
        A_eq.append(A_temp)
        #Optimization
        c[i:n**2:n] *= pos_bias[i]
    b_eq = np.ones(n*2)
    A_eq = np.asarray(A_eq)
    bounds = [(0,1) for _ in range(n**2)] + [(0,None) for _ in range(n_g+n_i)]


    A_ub = []
    b_ub = np.zeros(n_g+n_i)
    if(group_fair):
        U = []
        for group in G:
            #Avoid devision by zero
            u = np.max([sum(np.asarray(popularity)[group]), 0.01])
            U.append(u)
        comparisons = list(permutations(np.arange(len(G)),2))
        j = 0
        for a,b in comparisons:
            f = np.zeros(n_c)
            if len(G[a]) > 0 and len(G[b])>0:
                for i in range(n):
                    if impact:
                        tmp1 = popularity[i] / U[a] if i in G[a] else 0
                        tmp2 = popularity[i] / U[b] if i in G[b] else 0
                    else:
                        tmp1 = 1. / U[a] if i in G[a] else 0
                        tmp2 = 1. / U[b] if i in G[b] else 0
                    f[i*n:(i+1)*n] =  (tmp1 - tmp2) # * popularity[i] for equal impact instead of equal Exposure
                for i in range(n):
                    f[i:n**2:n] *= pos_bias[i]
                f[n**2+j] = -1
                if group_click_rel is not None:
                    b_ub[j] = LP_COMPENSATE_W * (group_click_rel[b] - group_click_rel[a])
            j += 1
            A_ub.append(f)

    if(ind_fair):
        comparisons = list(permutations(np.arange(len(popularity)),2))
        j = 0
        for a,b in comparisons:
            f = np.zeros(n_c)
            if(popularity[a] >= popularity[b]):
                tmp1 = 1. / np.max([0.01,popularity[a]])
                tmp2 = 1. / np.max([0.01,popularity[b]])
                f[a*n:(a+1)*n] = tmp1
                f[a*n:(a+1)*n] *= pos_bias
                f[b*n:(b+1)*n] = -1 *  tmp2
                f[b*n:(b+1)*n] *= pos_bias

                f[n**2+n_g+j] = -1
            j += 1
            A_ub.append(f)

    res = scipy.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, options=dict(bland =True, tol=1e-12), method = "interior-point")
    probabilistic_ranking = np.reshape(res.x[:n**2],(n,n))


    if(debug):
        print("Shape of the constrains", np.shape(A_eq), "with {} items and {} groups".format(n, len(G)))
        print("Fairness constraint:", np.round(np.dot(A_eq,res.x),4))
        #print("Constructed probabilistic_ranking with score {}: \n".format(res.fun), np.round(probabilistic_ranking,2))
        print("Col sum: ", np.sum(probabilistic_ranking,axis=0))
        print("Row sum: ", np.sum(probabilistic_ranking,axis=1))
        #plt.matshow(A_eq)
        #plt.colorbar()
        #plt.plot()
        plt.matshow(probabilistic_ranking)
        plt.colorbar()
        plt.plot()

    #Sample from probabilistic ranking using Birkhoff-von-Neumann decomposition
    try:
        decomp = birkhoff.birkhoff_von_neumann_decomposition(probabilistic_ranking)
    except:
        decomp = birkhoff.approx_birkhoff_von_neumann_decomposition(probabilistic_ranking)

        if debug:
            print("Could get a approx decomposition with {}% accuracy".format(100*sum([x[0] for x in decomp])) )
            #print(probabilistic_ranking)

    return decomp

def ideal_rank(users, item_affs, DATA_SET = 0):
    aff_prob = np.zeros(len(item_affs))
    for user in users:
        aff_prob += affinity_score(user, item_affs, DATA_SET=DATA_SET)

    return aff_prob, (np.argsort(aff_prob)[::-1])

def plot_topk_instant_group_size(top_k_list=[3,5,10,20,30],file="plots/EXP1_topk/Fairness_Data.npy",\
              labels=["1","2","3"],methods=["1","2","3"],PLOT_PREFIX="trial/",click_models=["pbm"]):
    data = np.load(file, allow_pickle=True)
    method_num=len(data)
    trial_num,iteration_num=data[0]["rank_hist"].shape[:2]
    G=data[0]['G']
    group_num=len(G[0])
    run_data=[]
    data_top_k={}
    
    for top_k in top_k_list:
        data_all_meth=[]
        for meth in range(method_num):
            data_meth_cur={}
            group_prop=np.zeros((trial_num,iteration_num,group_num))
            group_click=np.zeros((trial_num,iteration_num,group_num))
            true_group_rel=np.zeros((trial_num,iteration_num,group_num))
            pophist=data[meth]["pophist"]
            clicks=np.zeros_like(pophist)
            clicks[:,0,:]=pophist[:,0,:]
            clicks[:,1:,:]=pophist[:,1:,:]-pophist[:,:-1,:]
            
            for trial in range(trial_num):
                G_trial=data[meth]['G'][trial]
                rank_hist=data[meth]['rank_hist'][trial,:,:top_k]
                propensity=data[meth]['propensity_history'][trial,:,:]
                rele_gnd=data[meth]["numerical_relevance"][trial,:]
                clicks_trial=clicks[trial,:,:]
                group_num_each=np.array([len(i)for i in data[meth]['G'][trial]])
                true_rele_aver=data[meth]["true_rel"][trial,0]/group_num_each
                
#                 print(true_rele_aver,"true_rele_aver")
                for iteration in range(iteration_num):
                    propensity_cur=propensity[iteration,:]
                    G_cur=[]
                    top_k_ind=rank_hist[iteration,:]
                    click_cur=clicks_trial[iteration,:]
                    for j in G[trial]:
                        G_cur.append(np.intersect1d(j,top_k_ind).astype(int))
#                     print(G_cur)
                    for G_i in range(len(G_trial)):
                        cur_propensity=0
                        cur_click=0
                        if len(G_cur[G_i])==0:
#                             print(G_cur[G_i])
                            cur_propensity=0
                            cur_click=0
                        else:
                            cur_propensity=np.sum(propensity_cur[G_cur[G_i]])/len(G_cur[G_i])
                            cur_click=np.sum(click_cur[G_cur[G_i]])/len(G_cur[G_i])
                        group_prop[trial,iteration, G_i] = cur_propensity
    #                     print(group_prop[trial,iteration, :])
                        group_click[trial,iteration, G_i]=cur_click
#                     true_group_rel[trial,iteration, :] = [np.sum(rele_gnd[G_cur[g]]) for g in range(len(G_trial))]  
                    true_group_rel[trial,iteration, :] = [true_rele_aver[g] for g in range(len(G_trial))]
    
#                     group_prop[trial,iteration, :] = [np.sum(propensity_cur[G_cur[i]]) for i in range(len(G_trial))]
#                     group_click[trial,iteration, :]=[np.sum(click_cur[G_cur[i]]) for i in range(len(G_trial))]
# #                     true_group_rel[trial,iteration, :] = [np.sum(rele_gnd[G_cur[g]]) for g in range(len(G_trial))]  
#                     true_group_rel[trial,iteration, :] = [true_rele_aver[g]*len(G_cur[g]) for g in range(len(G_trial))]
#                     print(true_group_rel[trial,iteration, :])
#             print(np.cumsum(group_prop,axis=1).shape,"*")
            range_it=np.arange(1,iteration_num+1)[np.newaxis,:,np.newaxis]
            data_meth_cur["prop"]=np.cumsum(group_prop,axis=1)/range_it     
            data_meth_cur["true_rel"]=np.cumsum(true_group_rel ,axis=1)/range_it 
            data_meth_cur["clicks"]=np.cumsum(group_click ,axis=1)
            data_all_meth.append(data_meth_cur)    
#             return data_all_meth
        run_data=data_all_meth
        overall_fairness = np.zeros((len(click_models) * method_num, trial_num, iteration_num, 4))
        pair_group_combinations = [(a, b) for a in range(group_num) for b in range(a + 1, group_num)]
        for i, data_cur in enumerate(run_data):

            for a, b in pair_group_combinations:
                overall_fairness[i, :, :, 0] += np.abs(
                    data_cur["prop"][:, :, a] / data_cur["true_rel"][:, :, a] - data_cur["prop"][:, :, b] / data_cur["true_rel"][:, :, b])
                overall_fairness[i, :, :, 1] += np.abs(
                    data_cur["prop"][:, :, a] / data_cur["true_rel"][:, :, a] - data_cur["prop"][:, :, b] / data_cur["true_rel"][:, :, b])
                overall_fairness[i, :, :, 2] += np.abs(
                    data_cur["clicks"][:, :, a] / data_cur["true_rel"][:, :, a] - data_cur["clicks"][:, :, b] / data_cur["true_rel"][:, :, b])
                overall_fairness[i, :, :, 3] += np.abs(
                    data_cur["clicks"][:, :, a] / data_cur["true_rel"][:, :, a] - data_cur["clicks"][:, :, b] / data_cur["true_rel"][:, :,                                                                                            b])

        overall_fairness /= len(pair_group_combinations)
        PLOT_PREFIX_cur=PLOT_PREFIX+"unfairness_top"+str(top_k)+"/"
        if not os.path.exists(PLOT_PREFIX_cur):
            os.makedirs(PLOT_PREFIX_cur)
        plot_unfairness_over_time(overall_fairness, click_models, methods, \
                                  True,PLOT_PREFIX=PLOT_PREFIX_cur)


def plot_topk(top_k_list=[3,5,10,20,30],file="plots/EXP1_topk/Fairness_Data.npy",\
              labels=["1","2","3"],methods=["1","2","3"],PLOT_PREFIX="trial/",click_models=["pbm"]):
    data = np.load(file, allow_pickle=True)
    method_num=len(data)
    trial_num,iteration_num=data[0]["rank_hist"].shape[:2]
    G=data[0]['G']
    group_num=len(G[0])
    run_data=[]
    data_top_k={}
    
    for top_k in top_k_list:
        data_all_meth=[]
        for meth in range(method_num):
            data_meth_cur={}
            group_prop=np.zeros((trial_num,iteration_num,group_num))
            group_click=np.zeros((trial_num,iteration_num,group_num))
            true_group_rel=np.zeros((trial_num,iteration_num,group_num))
            pophist=data[meth]["pophist"]
            clicks=np.zeros_like(pophist)
            clicks[:,0,:]=pophist[:,0,:]
            clicks[:,1:,:]=pophist[:,1:,:]-pophist[:,:-1,:]
            
            for trial in range(trial_num):
                G_trial=data[meth]['G'][trial]
                rank_hist=data[meth]['rank_hist'][trial,:,:top_k]
                propensity=data[meth]['propensity_history'][trial,:,:]
                rele_gnd=data[meth]["numerical_relevance"][trial,:]
                clicks_trial=clicks[trial,:,:]
                group_num_each=np.array([len(i)for i in data[meth]['G'][trial]])
                true_rele_aver=data[meth]["true_rel"][trial,0]/group_num_each
                
#                 print(true_rele_aver,"true_rele_aver")
                for iteration in range(iteration_num):
                    propensity_cur=propensity[iteration,:]
                    G_cur=[]
                    top_k_ind=rank_hist[iteration,:]
                    click_cur=clicks_trial[iteration,:]
                    for j in G[trial]:
                        G_cur.append(np.intersect1d(j,top_k_ind).astype(int))
#                     print(G_cur)
                    for G_i in range(len(G_trial)):
                        cur_propensity=0
                        cur_click=0
                        if len(G_cur[G_i])==0:
#                             print(G_cur[G_i])
                            cur_propensity=0
                            cur_click=0
                        else:
                            cur_propensity=np.sum(propensity_cur[G_cur[G_i]])/len(G_trial[G_i])
                            cur_click=np.sum(click_cur[G_cur[G_i]])/len(G_trial[G_i])
                        group_prop[trial,iteration, G_i] = cur_propensity
    #                     print(group_prop[trial,iteration, :])
                        group_click[trial,iteration, G_i]=cur_click
#                     true_group_rel[trial,iteration, :] = [np.sum(rele_gnd[G_cur[g]]) for g in range(len(G_trial))]  
                    true_group_rel[trial,iteration, :] = [true_rele_aver[g] for g in range(len(G_trial))]
            range_it=np.arange(1,iteration_num+1)[np.newaxis,:,np.newaxis]
            data_meth_cur["prop"]=np.cumsum(group_prop,axis=1)/range_it     
            data_meth_cur["true_rel"]=np.cumsum(true_group_rel ,axis=1)/range_it 
            data_meth_cur["clicks"]=np.cumsum(group_click ,axis=1)
            data_all_meth.append(data_meth_cur)    
#             return data_all_meth
        run_data=data_all_meth
        overall_fairness = np.zeros((len(click_models) * method_num, trial_num, iteration_num, 4))
        pair_group_combinations = [(a, b) for a in range(group_num) for b in range(a + 1, group_num)]
        for i, data_cur in enumerate(run_data):

            for a, b in pair_group_combinations:
                overall_fairness[i, :, :, 0] += np.abs(
                    data_cur["prop"][:, :, a] / data_cur["true_rel"][:, :, a] - data_cur["prop"][:, :, b] / data_cur["true_rel"][:, :, b])
                overall_fairness[i, :, :, 1] += np.abs(
                    data_cur["prop"][:, :, a] / data_cur["true_rel"][:, :, a] - data_cur["prop"][:, :, b] / data_cur["true_rel"][:, :, b])
                overall_fairness[i, :, :, 2] += np.abs(
                    data_cur["clicks"][:, :, a] / data_cur["true_rel"][:, :, a] - data_cur["clicks"][:, :, b] / data_cur["true_rel"][:, :, b])
                overall_fairness[i, :, :, 3] += np.abs(
                    data_cur["clicks"][:, :, a] / data_cur["true_rel"][:, :, a] - data_cur["clicks"][:, :, b] / data_cur["true_rel"][:, :,                                                                                            b])

        overall_fairness /= len(pair_group_combinations)
        PLOT_PREFIX_cur=PLOT_PREFIX+"unfairness_top"+str(top_k)+"/"
        if not os.path.exists(PLOT_PREFIX_cur):
            os.makedirs(PLOT_PREFIX_cur)
        plot_unfairness_over_time(overall_fairness, click_models, methods, \
                                  True,PLOT_PREFIX=PLOT_PREFIX_cur)
# Function that simulates and monitor the convergence to the relevance + the developement of cummulative fairness
@ex.capture
def collect_relevance_convergence(items, start_popularity, trials=10, methods=["Naive", "IPS"],
                                  click_models=["PBM_log"], iterations=2000, plot_individual_fairness=True,
                                  multiple_items=None, PLOT_PREFIX="", MOVIE_RATING_FILE="",top_k_list=[3,5,10,20,30],add_args=None):

    global get_numerical_relevances
    add_args=copy.deepcopy(add_args)
    rel_diff = []
    if multiple_items is None:
        G = assign_groups(items)
    else:
        if multiple_items == -1:
            G = assign_groups(items)
        else:
            assert (len(multiple_items) == trials)
            G = assign_groups(multiple_items[0])
    print(type(click_models),type(methods),type(trials),type(iterations))
    overall_fairness = np.zeros((len(click_models) * len(methods), trials, iterations, 4))
    pair_group_combinations = [(a, b) for a in range(len(G)) for b in range(a + 1, len(G))]
    count = 0
    run_data = []
    frac_c = [[] for i in range(len(G))]
    if len(items)>30:
        top_k_list=top_k_list+[50,100]
    print(top_k_list,"top_k_list")
    add_args["top_k_list"]=top_k_list
    nn_errors = []
    item_length=len(items)
    method_dict = {"Naive": "Naive", \
#                    "IPS": r'$\hat{R}^{IPS}(d)$', \
                    "IPS": "D_ULTR(Glob)",\
                   "Pers": "D-ULTR", "Skyline-Pers": "Skyline",
                   "Fair-I-IPS": "FairCo(Imp)", "Fair-E-IPS": "FairCo(Exp)", "Fair-I-Pers": "FairCo(Imp)",
                   "Fair-E-Pers": "FairCo(Exp)", "Fair-I-IPS-LP": "LinProg(Imp)", "Fair-E-IPS-LP": "LinProg","IPS_I_top_down":"IPS_I_top_down","IPS_E_top_down":"IPS_E_top_down","Fair-I_top_down-Pers":"Fair-I_top_down-Pers","Fair-E_top_down-Pers":"Fair-E_top_down-Pers","IPS_E_greedy":"Fair_MMR","Fair-E_greedy-Pers":"Fair-E_greedy-Pers","Fair-E-IPS_pers-LP":"Fair-E-IPS_pers-LP"}
    user_generator = None
    for click_model in click_models:

        if "lambda" in click_model and "lambda_cutoff" not in click_model: #For vcomparing different Lambdas,
            lam = float(click_model.replace("lambda", ""))
            ex.add_config({
                'KP': lam,
                'W_FAIR': lam
            })
            click_model = "PBM_log"
        elif "lambda_cutoff" in click_model: #For vcomparing different Lambdas,
            lam = float(click_model.replace("lambda_cutoff", ""))
            ex.add_config({
                'KP': lam,
                'W_FAIR': lam
            })
            click_model = "PBM_cutoff"
        for method in methods:
            start_time = time.time()
            rel_diff_trial = []
            fairness = {"prop": np.zeros((trials, iterations, len(G))),
                        "clicks": np.zeros((trials, iterations, len(G))), "rel": np.zeros((trials, iterations, len(G))),
                        "true_rel": np.zeros((trials, iterations, len(G))), "NDCG": np.zeros((trials, iterations)),\
            "aff_scores": np.zeros((trials, iterations,item_length)),\
                        "pophist":np.zeros((trials, iterations,item_length)),"hist":np.zeros((trials, iterations,item_length)),"w_pophist":np.zeros((trials, iterations,item_length)),\
                        "propensity_history":np.zeros((trials, iterations,item_length)),"G":[None]*trials,"rank_hist":np.zeros((trials, iterations,item_length)),\
                        "numerical_relevance":np.zeros((trials, item_length)),\
                        "rank_group_id_sorted":np.zeros((trials, iterations,item_length)),\
                       "rank_group_split":np.zeros((trials, len(G),item_length)),
                       "rank_group_split_click":np.zeros((trials, len(G),item_length))}
            for i in top_k_list:
                fairness["NDCG_"+str(i)]=np.zeros((trials, iterations))
            nn_error_trial = []
            for i in range(trials):
                print("trails",i,"method",method)
                if multiple_items is not None:
                    if multiple_items == -1:  # Load a new bernully relevance table
                        MOVIE_RATING_FILE = MOVIE_RATING_FILE.replace("trial{}.npy".format(i-1),"trial{}.npy".format(i))
                        #MOVIE_RATING_FILE = "data/movie_data_binary_latent_5Comp_trial{}.npy".format(i)
                        user_generator = Usersampler(MOVIE_RATING_FILE=MOVIE_RATING_FILE)
                        ranking, _, _ = data_utils.load_movie_data_saved(MOVIE_RATING_FILE)
                        get_numerical_relevances = lambda x: np.mean(ranking, axis=0)

                    else:
                        items = multiple_items[i]
                        G = assign_groups(items)
                popularity = np.copy(start_popularity)
                # Run Simulation
                iterations, ranking_hist, popularity_hist, final_ranking, users, ideal, mean_relevances, w_pophist, errors, mean_exposure, fairness_hist, p_pophist = \
                    simulate(popularity, items, ranking_method=method, click_model=click_model, iterations=iterations, user_generator=user_generator,add_args=add_args)
                ranking_hist = ranking_hist.astype(int)
                if "Pers" in method:
                    nn_error_trial.append(errors)

                # Calculate the relevance difference between true relevance and approximation
                # Diff = |rel' - rel|
                if method == "Naive":
                    rel_estimate = popularity_hist / np.arange(1, iterations + 1)[:, np.newaxis]
                elif "Pers" in method:
                    p_pophist[99:, :] = [np.sum(p_pophist[98:100 + i, :], axis=0) for i in range(len(p_pophist) - 99)]

                    rel_estimate = p_pophist / (np.arange(iterations) + 1)[:, np.newaxis]
                else:
                    rel_estimate = w_pophist / np.arange(1, iterations + 1)[:, np.newaxis]

                rel_diff_trial.append(np.mean(np.abs(rel_estimate - (mean_relevances)[np.newaxis, :]), axis=1))

                # Cummulative Fairness per Iteration summed over trials
                print(fairness_hist.keys(),"vfairness_hist.keys()")
                for key, value in fairness_hist.items():
                    fairness[key][i] = value
                if (trials <= 1):
                    # Plot Group Clicks and Items Average Rank
                    group_item_clicks(popularity_hist[-1], G)
                    plot_average_rank(ranking_hist, G)
                    print("Relevance Difference: ", np.sum((mean_relevances - rel_estimate[-1]) ** 2))

                    # Plot Ranking History
                    plt.title("Ranking History")
                    plt.axis([0, iterations, 0, len(items)])
                    if len(G) <= 3:
                        group_colors = {0: "blue", 1: "red", 2: "black"}
                        group_labels = {0: "Negative", 1: "Positive", 2: "black"}
                    else:
                        group_colors = [None for i in range(len(G))]
                    item_rank_path = np.ones((iterations, len(items)))
                    for i in range(iterations):
                        item_rank_path[i, ranking_hist[i, :]] = np.arange(len(items))
                    for i in range(len(items)):
                        group_color_i = group_colors[[x for x in range(len(G)) if i in G[x]][0]]
                        plt.plot(np.arange(iterations), item_rank_path[:, i], color=group_color_i)

                    custom_lines = [Line2D([0], [0], color="blue", lw=4),
                                    Line2D([0], [0], color="red", lw=4)]

                    plt.legend(custom_lines, ['Negative', 'Positive'])
                    plt.legend()
                    plt.savefig(PLOT_PREFIX + "Rankinghistory_" + click_model + "_" + method + ".pdf",
                                bbox_inches="tight")

            print("Time for " + click_model + " " + method + " was: {0:.4f}".format(time.time() - start_time))

            if "Pers" in method:
                mean_trial_error = np.mean(np.asarray(nn_error_trial), axis=0)
                nn_errors.append(mean_trial_error)

            count += 1
            # Collect Data for later
            run_data.append(fairness)

            for i in range(len(G)):
                frac_c[i].append(np.mean(fairness["clicks"][:, -1, i]) / iterations)

            if (len(rel_diff_trial) == 1):
                rel_tmp = np.asarray(rel_diff_trial[0])
                rel_std = np.zeros(np.shape(rel_tmp))
            else:
                rel_tmp = np.mean(np.asarray(rel_diff_trial), axis=0)
                rel_std = np.std(np.asarray(rel_diff_trial), axis=0)
            rel_diff.append([rel_tmp, method_dict[method], rel_std])

    np.save(PLOT_PREFIX + "Fairness_Data.npy", run_data)

    # Plot NDCG
    plt.figure("NDCG")
    # plt.title("Average NDCG")
    # labels = [ a + "\n" + b for a in click_models for b in methods]
    labels = [b for a in click_models for b in methods]
    for i, nd in enumerate(run_data):
        plot_ndcg(np.mean(nd["NDCG"], axis=0), label=labels[i], plot=False, window_size=100, std=nd["NDCG"])
    plt.legend()
    ax = plt.gca()
    plt.savefig(PLOT_PREFIX + "NDCG.pdf", bbox_inches="tight", dpi=800)
    plt.show()
    plt.close("all")

    # Plot Clicks
    plot_click_bar_plot(frac_c, labels, save=True)

    if True:
        plt.close("all")
        # Plot Convergence of Relevance
        for y in rel_diff:
            p = plt.plot(np.arange(len(y[0])), y[0], label=y[1])
            color = p[-1].get_color()
            plt.fill_between(np.arange(len(y[0])), y[0] - y[2],
                             y[0] + y[2], alpha=0.4, color=color)

        plt.legend(loc="best")
        plt.axis([0, len(y[0]), 0, 0.3])
        # plt.ylabel("Avg diff between \n True & Estimated Relevance  ")
        plt.ylabel(r'average $|\hat{R}(d) - {R}(d)|$')

        plt.xlabel("Users")
        plt.savefig(PLOT_PREFIX + "Relevance_convergence.pdf", bbox_inches="tight")
        plt.show()

    plot_neural_error(nn_errors, [b for a in click_models for b in methods if "Pers" in b])
    # Plot Unfairness over time between different models

    for i, data in enumerate(run_data):

        for a, b in pair_group_combinations:
            overall_fairness[i, :, :, 0] += np.abs(
                data["prop"][:, :, a] / data["rel"][:, :, a] - data["prop"][:, :, b] / data["rel"][:, :, b])
            overall_fairness[i, :, :, 1] += np.abs(
                data["prop"][:, :, a] / data["true_rel"][:, :, a] - data["prop"][:, :, b] / data["true_rel"][:, :, b])
            overall_fairness[i, :, :, 2] += np.abs(
                data["clicks"][:, :, a] / data["rel"][:, :, a] - data["clicks"][:, :, b] / data["rel"][:, :, b])
            overall_fairness[i, :, :, 3] += np.abs(
                data["clicks"][:, :, a] / data["true_rel"][:, :, a] - data["clicks"][:, :, b] / data["true_rel"][:, :,
                                                                                                b])

    overall_fairness /= len(pair_group_combinations)
    plot_unfairness_over_time(overall_fairness, click_models, methods, True)
#     print(labels,methods,click_models,"")
    plot_topk(top_k_list=top_k_list,file=PLOT_PREFIX + "Fairness_Data.npy",\
              labels=labels,methods=methods,PLOT_PREFIX=PLOT_PREFIX,click_models=click_models)
    ndcg_full = []
    for data in run_data:
        ndcg_full.append(data["NDCG"])
    plt.close('all')
    combine_and_plot_ndcg_unfairness(ndcg_full,overall_fairness[:, :, :, 1],labels= labels, selection=np.arange(len(run_data)), name=PLOT_PREFIX + "NDCG_UnfairExposure.pdf",type = 0 )
    combine_and_plot_ndcg_unfairness(ndcg_full,overall_fairness[:, :, :, 3],labels= labels, selection=np.arange(len(run_data)), name=PLOT_PREFIX + "NDCG_UnfairImpact.pdf",type = 1 )