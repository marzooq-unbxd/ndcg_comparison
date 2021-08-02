import urllib

import requests
import os
import pandas as pd
import json

from tqdm import tqdm

api_key='c28b3b5ae91e7b48bf78825e7b63b483'
isite_name='globalindustrial-com702401520254089'
SEARCH_URL_false = "http://search.unbxd.io/{}/{}/search?fl=product_id,score&analytics=false&qis=false&debug=true&debug.structured=true".format(api_key,isite_name)
categories_pred = {"Cleaning Supplies>Bathroom Cleaners": 25}
MIN_BOOST_PARAMETER = 10
MAX_BOOST_PARAMETER = 40

score_map = {"clicks": 1, "carts": 8, "orders": 20}

# config_manager = configService.AlbusConfigManager()
# config_manager.register_client()
# statistics = config_manager.get_config(isite_name, 'qcs', 'statistical', 'deployments')
# statistical_category = statistics[0]["model_metadata"]["statistical_category"]+"_uFilter"
# min_score_threshold = statistics[0]["model_metadata"]["min_score_threshold"]
# num_prod_threshold = statistics[0]["model_metadata"]["num_prod_threshold"]
num_prod_threshold= [
      100,
      200,
      400
    ]
min_score_threshold=20
statistical_category= "categoryPathName"


def retro_dictify(row):
    t_dict = {}
    d = {}
    #for row in frame:
    cat_dict = {}
    for cat,sc  in (zip(row[:-1:2],row[1:len(row)-1:2])): #row[:-1:2]
        #cat = row[i]
        #sc = row[i + 1]
        if cat != 'na':

            cat_dict[cat] = sc
        else:
            break
    d['categories'] = cat_dict
        #t_dict[tq] = d
    return d

def applyFilterStrategy(x):
    statisticalData=retro_dictify(list(x[[ 'Category1', 'Score1', 'Category2', 'Score2',
       'Category3', 'Score3', 'Category4', 'Score4', 'Category5', 'Score5']]))
    numberOfProducts = x['numberOfProducts']
    maxCategoryScore = max([statisticalData['categories'][category] for category in statisticalData['categories'].keys()])
    scoring_list=[statisticalData['categories'][category] for category in statisticalData['categories'].keys()]
    x['filter'],x['thresh']= False, None

    if not numberOfProducts:
        #If we don't have number of products information,
        #  filter only if maxCategoryScore is quite high, else boost
        if maxCategoryScore > 3 * min_score_threshold:
            cur_thresh_score = get_current_thresold_score(0.05, scoring_list, min_score_threshold)
            x['filter'],x['thresh']=True, cur_thresh_score # if current score is < 0.05* cur agg of scores
            #return x
        else:
            x['filter'],x['thresh']= False, min_score_threshold
            #return x

    elif maxCategoryScore > 2 * min_score_threshold and numberOfProducts > num_prod_threshold[0]:
        cur_thresh_score = get_current_thresold_score(0.05, scoring_list, min_score_threshold)
        x['filter'],x['thresh']= True, cur_thresh_score
        #return x

    elif maxCategoryScore > 1.5 * min_score_threshold and numberOfProducts > num_prod_threshold[1]:
        x['filter'],x['thresh']= True, min_score_threshold
        #return x

    elif maxCategoryScore > min_score_threshold and numberOfProducts > num_prod_threshold[2]:
        x['filter'],x['thresh']= True, 0.9 * min_score_threshold
        #return x

    # If number of products is not that huge and we want to just boost
    elif maxCategoryScore > 2 * min_score_threshold:
        cur_thresh_score = get_current_thresold_score(0.1, scoring_list, min_score_threshold)
        x['filter'],x['thresh']= False, cur_thresh_score
        #return x

    elif maxCategoryScore > min_score_threshold:
        x['filter'],x['thresh']= False, min(min_score_threshold, 0.8 * maxCategoryScore)
        #return x

    x['filtered_categories']=None
    x['boosted_categories']=None
    if x['filter']:
        #fq=categoryPath_uFilter:"Cables>Audio Cables>Auxiliary Cables"
        # filtered_categories = " OR ".join(
        #     [statistical_category + ":" + '\"' + key + '\"' for key in statisticalData['categories'].keys() if
        #      statisticalData['categories'][key] > min_score_threshold])
        filtered_categories = " OR ".join(
            [statistical_category + "_uFilter:" + '"' + key + '"' for key in statisticalData['categories'].keys() if
             statisticalData['categories'][key] > min_score_threshold])
        fq_parameter_toadd="&fq="+filtered_categories
        solr_parameters_added=SEARCH_URL_false+"&q={}".format(x.Original_Head_Query)+fq_parameter_toadd
        #solr_parameters_added+="&rows=" + str(prod_per_page)

        x['filtered_categories'] = solr_parameters_added
    elif x['thresh']:
        #boostedCategories = list()
        boostedCategories_variables_to_add = ""
        bfparameter_each_cat =[]
        boost_count_category=0
        categories = statisticalData['categories']
        for category in categories.keys():
            score = categories[category]
            if score < x['thresh']:
                continue

            boostedCategories_variables_to_add+=f"&xyz{boost_count_category}=categoryPathName_uFilter:\"{category}\""
            #boostedCategories+="&boost=if(and(gt(query($bf1),0)),{},0)".format(boost_value_test)
            bfparameter_each_cat.append("if(and(gt(query($xyz{}),0)),{},0)".format(boost_count_category,MIN_BOOST_PARAMETER+(score/maxCategoryScore)*(MAX_BOOST_PARAMETER-MIN_BOOST_PARAMETER)))
            ###todo edit boost_value_test
            boost_count_category+=1
            #"category_path" : category, "confidence" : score/maxCategoryScore
        bfparameter_to_add = "&bf=max("+','.join(w for w in bfparameter_each_cat)+")"
        solr_parameters_added=SEARCH_URL_false+"&q={}".format(x.Original_Head_Query)+boostedCategories_variables_to_add+bfparameter_to_add
        #solr_parameters_added+="&rows=" + str(prod_per_page)
        # for cat in categories_pred:
        #     category_param="&bf1=categoryPathName_uFilter:{}"
        #     boost_param="&boost=if(and(gt(query($bf1),0)),{},0)".format(boost_value_test)
        #     SEARCH_URL_qcp_res+=category_param
        #     SEARCH_URL_qcp_res +=boost_param
        x["boosted_categories"] = solr_parameters_added

    return x
def applyFilterStrategy_one_query(qu,statisticalData,prod_per_page):
    maxCategoryScore = max([statisticalData['categories'][category] for category in statisticalData['categories'].keys()])
    scoring_list=[statisticalData['categories'][category] for category in statisticalData['categories'].keys()]
    filt,thres= False, None
    numberOfProducts = statisticalData['numberOfProducts']
    if not numberOfProducts:
        #If we don't have number of products information,
        #  filter only if maxCategoryScore is quite high, else boost
        if maxCategoryScore > 3 * min_score_threshold:
            cur_thresh_score = get_current_thresold_score(0.05, scoring_list, min_score_threshold)
            filt,thres=True, cur_thresh_score # if current score is < 0.05* cur agg of scores
            #return x
        else:
            filt,thres= False, min_score_threshold
            #return x

    elif maxCategoryScore > 2 * min_score_threshold and numberOfProducts > num_prod_threshold[0]:
        cur_thresh_score = get_current_thresold_score(0.05, scoring_list, min_score_threshold)
        filt,thres= True, cur_thresh_score
        #return x

    elif maxCategoryScore > 1.5 * min_score_threshold and numberOfProducts > num_prod_threshold[1]:
        filt,thres= True, min_score_threshold
        #return x

    elif maxCategoryScore > min_score_threshold and numberOfProducts > num_prod_threshold[2]:
        filt,thres= True, 0.9 * min_score_threshold
        #return x

    # If number of products is not that huge and we want to just boost
    elif maxCategoryScore > 2 * min_score_threshold:
        cur_thresh_score = get_current_thresold_score(0.1, scoring_list, min_score_threshold)
        filt,thres= False, cur_thresh_score
        #return x

    elif maxCategoryScore > min_score_threshold:
        filt,thres= False, min(min_score_threshold, 0.8 * maxCategoryScore)
        #return x

    # x['filtered_categories']=None
    # x['boosted_categories']=None
    if filt:
        #fq=categoryPath_uFilter:"Cables>Audio Cables>Auxiliary Cables"
        # filtered_categories = " OR ".join(
        #     [statistical_category + ":" + '\"' + key + '\"' for key in statisticalData['categories'].keys() if
        #      statisticalData['categories'][key] > min_score_threshold])
        filtered_categories = " OR ".join(
            [urllib.parse.quote(statistical_category + "_uFilter:" + '"' + key + '"') for key in statisticalData['categories'].keys() if
             statisticalData['categories'][key] > min_score_threshold])
        fq_parameter_toadd="&fq="+filtered_categories
        solr_parameters_added=SEARCH_URL_false+"&q={}".format(qu)+fq_parameter_toadd
        solr_parameters_added+="&rows=" + str(prod_per_page)

        return solr_parameters_added,filt
    elif thres:
        #boostedCategories = list()
        boostedCategories_variables_to_add = ""
        bfparameter_each_cat =[]
        boost_count_category=0
        categories = statisticalData['categories']
        for category in categories.keys():
            score = categories[category]
            if score < thres:
                continue
            catiiii = urllib.parse.quote(f"categoryPathName_uFilter:\"{category}\"")
            boostedCategories_variables_to_add+=f"&xyz{boost_count_category}={catiiii}"
            #boostedCategories+="&boost=if(and(gt(query($bf1),0)),{},0)".format(boost_value_test)
            bfparameter_each_cat.append("if(and(gt(query($xyz{}),0)),{},0)".format(boost_count_category,MIN_BOOST_PARAMETER+(score/maxCategoryScore)*(MAX_BOOST_PARAMETER-MIN_BOOST_PARAMETER)))
            ###todo edit boost_value_test
            boost_count_category+=1
            #"category_path" : category, "confidence" : score/maxCategoryScore
        bfparameter_to_add = "&bf=max("+','.join(w for w in bfparameter_each_cat)+")"
        solr_parameters_added=SEARCH_URL_false+"&q={}".format(qu)+boostedCategories_variables_to_add+bfparameter_to_add
        solr_parameters_added+="&rows=" + str(prod_per_page)

        # for cat in categories_pred:
        #     category_param="&bf1=categoryPathName_uFilter:{}"
        #     boost_param="&boost=if(and(gt(query($bf1),0)),{},0)".format(boost_value_test)
        #     SEARCH_URL_qcp_res+=category_param
        #     SEARCH_URL_qcp_res +=boost_param
        return solr_parameters_added,filt

    #return x
def get_current_thresold_score(percent, score_list, min_score_threshold):
    score_list.sort(reverse=True)
    agg=score_list[0]
    delta = 0.1
    for sc in score_list[1:]:
        if sc>=percent*agg:
            agg+=sc
        else:
            return max(min_score_threshold, sc + delta)
    return min_score_threshold


def get_score(clicks, carts, orders):
    return clicks * score_map["clicks"] + carts * score_map["carts"] + orders * score_map["orders"]


if __name__ == '__main__':

    with open('query_files/globalindustrial-com702401520254089/hq/statisticalData.json', 'r') as f:

        stat_all = json.load(f)
    from pandarallel import pandarallel
    query_cco =pd.read_csv('querycco.csv')

    pandarallel.initialize(nb_workers=3, progress_bar=True)
    query_cco['agg_score'] = query_cco.parallel_apply(lambda row: get_score(row['clicks'], row['carts'], row['orders']),
                                                      axis=1)
    query_cco.drop(columns=['orders','carts','clicks'],inplace=True)
    query_cco.to_csv('querycco1.csv')
   
