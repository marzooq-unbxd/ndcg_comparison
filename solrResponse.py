from multiprocessing.pool import ThreadPool as TPool
from functools import partial, reduce

from pandarallel import pandarallel
from ratelimiter import RateLimiter
import pandas as pd
import sys
import requests
import urllib
import tqdm
from operator import itemgetter
from sklearn.metrics import ndcg_score,dcg_score
import numpy as np
from search_api_testing import applyFilterStrategy_one_query

isite_name =  "globalindustrial-com702401520254089"
api_key = "c28b3b5ae91e7b48bf78825e7b63b483"
import numpy as np
def discountedCumulativeGain(result):
    dcg = []
    for idx, val in enumerate(result):
        numerator = 2**val - 1
        # add 2 because python 0-index
        denominator =  np.log2(idx + 2)
        score = numerator/denominator
        dcg.append(score)
    return sum(dcg)
def normalizedDiscountedCumulativeGain(result, sorted_result):
    dcg = discountedCumulativeGain(result)
    idcg = discountedCumulativeGain(sorted_result)
    ndcg = dcg / idcg
    return ndcg
def kLargest(A,B, k):
    all_products = list(set(A).union(set(B)))
    all_products = [item[1] for item in all_products]

    all_products.sort(reverse = True)
    ans=[]
    for i in range(min(k,len(all_products))):
        ans.append (all_products[i])
    if len(ans)<k:
        ans.extend([0]*(k-len(ans)))
    pass
    return ans

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def recursive_get(d, *keys):
    return reduce(lambda c, k: c.get(k, {}), keys, d)

def get_cco(qu,pid):
    ans=query_cco[(query_cco['productId'] == pid) & (query_cco['query'] == qu)]    #ans= query_cco[(query_cco['productId']==pid)&(query_cco['query']==qu)]
    if ans.shape[0]==0:
        return 0  #if not present relvencae 0
    else:
        return ans['agg_score'].iloc[0]
    #return query_cco['productId']==pid#[statistical_category + '_processed']


def process_single_request(each_query, api_key, site_key, prod_per_page, timeout=.1, retry=False):
    encoded_query = urllib.parse.quote(each_query)
    control_query_url = f'''http://search.unbxd.io/{api_key}/{site_key}/search?q={encoded_query}&fl=product_id,score&rows={prod_per_page}&analytics=false&version=V2&qis=false'''
    #qcs_query_url = f'''http://search.unbxd.io/{api_key}/{site_key}/search?q={encoded_query}&fl=product_id,score&rows={10}&analytics=false&version=V2&qis=true'''
    qcs_query_url,filt =applyFilterStrategy_one_query(each_query,stat_all[each_query],prod_per_page)
    try:
        control_response = requests.get(control_query_url, timeout=timeout)
        qcs_response = requests.get(qcs_query_url, timeout=timeout)
    except Exception as timeOut:
        #print(timeOut)
        return {'q': each_query, 'data': None}
    if control_response.status_code == 200 and  qcs_response.status_code == 200:

            control_data = control_response.json()
            qcs_data = qcs_response.json()

            control_num_prod = recursive_get(control_data, "response", "numberOfProducts")
            qcs_num_prod = recursive_get(qcs_data, "response", "numberOfProducts")


            control_pid_list = [k.get('uniqueId') for k in recursive_get(control_data, "response", "products")]

            control_pid_cco = [(x ,get_cco(each_query,x)) for x in control_pid_list]
            control_pid_cco = [(x ,y) for x,y in control_pid_cco if y is not None]
            control_cco = [item[1] for item in control_pid_cco]
            if len(control_cco)<prod_per_page:
                control_cco.extend([0] * (prod_per_page - len(control_cco)))

            qcs_pid_list = [k.get('uniqueId') for k in recursive_get(qcs_data, "response", "products")]
            qcs_pid_cco = [(x ,get_cco(each_query,x)) for x in qcs_pid_list]

            qcs_cco = [item[1] for item in qcs_pid_cco]


            if len(qcs_cco)<prod_per_page:
                qcs_cco.extend([0] * (prod_per_page - len(qcs_cco)))

            #best_relevance_list = kLargest(qcs_cco,control_cco,prod_per_page)
            best_relevance_list = kLargest(qcs_pid_cco,control_pid_cco,prod_per_page)

            # if not any(sc !=0 for sc in control_cco):
            #     return {'q': each_query, 'data': None}
            if not any(sc !=0 for sc in qcs_cco) and not any(sc !=0 for sc in control_cco):
                    return {'q': each_query, 'data': None}
                # return {'q': each_query, 'data': (
                # each_query, 0,0, control_cco,qcs_cco , control_num_prod,
                # qcs_num_prod)}
            else:

                # ndcg_control = ndcg_score(
                #     np.array(best_relevance_list).reshape(1, -1), np.array(control_cco).reshape(1, -1))
                # ndcg_qcs = ndcg_score(
                # np.array(best_relevance_list).reshape(1,-1), np.array(qcs_cco).reshape(1,-1))
                ndcg_control = normalizedDiscountedCumulativeGain(control_cco,best_relevance_list)
                ndcg_qcs = normalizedDiscountedCumulativeGain(qcs_cco,best_relevance_list)

            #return {'q': each_query, 'data': (each_query, control_pid_list,control_num_prod,qcs_pid_list,qcs_num_prod)}
            return {'q': each_query, 'data': (each_query, ndcg_control,ndcg_qcs, control_cco,qcs_cco,control_num_prod,qcs_num_prod )}

    else:
        return {'q': each_query, 'data': None}


def process_query_df(query_list, site_key, api_key, rate_limit, prod_per_page):
    pool = TPool(rate_limit)
    uniq_quries = list(set([x.lower().strip() for x in query_list]))
    rate_limiter = RateLimiter(max_calls=1, period=1)
    data_agg = []
    failed_query = []
    for i, batch_quries in enumerate(tqdm.tqdm(batch(uniq_quries, rate_limit))):
        with rate_limiter:
            ret = pool.map(
                partial(
                    process_single_request,
                    api_key=api_key,
                    site_key=site_key,
                    prod_per_page=prod_per_page,
                    timeout=1.0,
                    retry=False
                ),
                batch_quries
            )
            data_agg.extend([r.get('data') for r in ret if isinstance(r.get('data'), tuple)])
            failed_query.extend([r.get('q') for r in ret if r.get('data') == 'retry'])
        if i % 100 and len(failed_query) > 0:
            with rate_limiter:
                ret = pool.map(
                    partial(
                        process_single_request,
                        api_key=api_key,
                        site_key=site_key,
                        prod_per_page=prod_per_page,
                        timeout=1.5,
                        retry=False
                    ),
                    failed_query
                )
                data_agg.extend([r.get('data') for r in ret if r.get('data')])
                failed_query = []

    fail_count = len(uniq_quries) - len(data_agg)
    df = pd.DataFrame(data_agg, columns="query,ndcg_control,ndcg_qcs, control_pid_list,qcs_pid_list,control_num_prod,qcs_num_prod".split(','))

    return df, fail_count

import json
def get_score(clicks, carts, orders):
    return clicks * score_map["clicks"] + carts * score_map["carts"] + orders * score_map["orders"]
score_map = {"clicks": 1, "carts": 8, "orders": 20}

with open('query_files/globalindustrial-com702401520254089/hq/statisticalData.json', 'r') as f:
    stat_all = json.load(f)
#q_list = ['execut desk paper trai', 'global drum storag cabinet']
q_list = ['singl pedest teacher desk','singl phase air compressor','simplehuman round open top can, 30 gallon brush ss cw1471']
#test_actual = pd.read_csv('test_queries/globalindustrial-com702401520254089/cleaned_hq_test.csv')
#q_list = list(test_actual['Stemmed_Head_Query'].head(25))

query_cco = pd.read_csv('querycco1.csv')
# pandarallel.initialize(progress_bar=True,nb_workers=3)
# query_cco['score'] = query_cco.parallel_apply(lambda row: get_score(row['clicks'], row['carts'], row['orders']),
#                                               axis=1)
q_list=list(stat_all.keys())[0:20000]
#q_list =[ 'master magnet ceram round base magnet rb20ccerbx 11 lbs. pull']
#q_list = ['huski rack & wire pallet rack post protector 24"h']
#answer = process_single_request('marbleiz top ergonom mat 3 foot wide cut blue', api_key, isite_name, 5, timeout=1, retry=False)

df_5, fail = process_query_df(query_list=q_list, site_key=isite_name, api_key=api_key, rate_limit=30, prod_per_page=5)
#
# df_10, fail = process_query_df(query_list=q_list, site_key=isite_name, api_key=api_key, rate_limit=30, prod_per_page=10)
pass
print(5)
df_5.to_csv('ndcg_results.csv')
