import numpy as np
import pandas as pd

import base_methods
from base_methods import column_formatter

import graph_methods
from graph_methods import network_by_date, load_vertices, find_p1_affiliations, load_edges, make_graph

from turicreate import SFrame, SGraph, SArray, load_sgraph, aggregate

def feature_creation(model_labels, list_of_graphs, p1_companies_uuid, radius=3):
    turicreate.config.set_runtime_config('TURI_DEFAULT_NUM_PYLAMBDA_WORKERS', 96)
    list_of_frames = []
    for idx,graph in enumerate(list_of_graphs):
            
        # CREATE SUBGRAPH
        print('Creating graph {}'.format(sgraph_idx[idx].upper()))
        smol_graph = graph.get_neighborhood(ids=model_labels, radius=radius, full_subgraph=True)   
        
        # FUNCTION FOR PAGERANK
        print('HERE_PR')
        DF_PG = add_pagerank(smol_graph, model_labels, idx)
        print(DF_PG.columns.to_list())
        if idx==1:
            DF_PG_1 = DF_PG
        if DF_PG.shape[0] != 0:
            list_of_frames.append(DF_PG)
        
        # FUNCTION FOR WEIGHTED PAGERANK
        print('HERE_PR_W')
        DF_PG_W = add_weighted_pagerank(smol_graph, model_labels, idx)
        print(DF_PG_W.columns.to_list())
        if idx==1:
            DF_PG_W_1 = DF_PG_W
        if DF_PG_W.shape[0] != 0:
            list_of_frames.append(DF_PG_W)
        
        # FUNCTION FOR SHORTEST PATH TOP 5
        print('HERE_SP')
        if idx==3:
            DF = add_shortest_path(smol_graph, model_labels, idx, DF_PG_1, p1_companies_uuid)
        else:
            DF = add_shortest_path(smol_graph, model_labels, idx, DF_PG, p1_companies_uuid)
        print(DF.columns.to_list())
        if DF.shape[0] != 0:
            list_of_frames.append(DF)
        
        # FUNCTION FOR SHORTEST PATH TOP 5 WEIGHTED
        print('HERE_SP_W')
        if idx==3:
            DF = add_weighted_shortest_path(smol_graph, model_labels, idx, DF_PG_W_1, p1_companies_uuid)
        else:
            DF = add_weighted_shortest_path(smol_graph, model_labels, idx, DF_PG_W, p1_companies_uuid)
        print(DF.columns.to_list())
        if DF.shape[0] != 0:
            list_of_frames.append(DF)
        
        # FUNCTION FOR K-CORE DECOPOSITION
        print('HERE_KC')
        DF = add_kcore(smol_graph, model_labels, idx)
        print(DF.columns.to_list())
        if DF.shape[0] != 0:
            list_of_frames.append(DF)
        
        # FUNCTION FOR DEGREES
        print('HERE_D')
        DF = add_degree(smol_graph, model_labels, idx)
        print(DF.columns.to_list())
        if DF.shape[0] != 0:
            list_of_frames.append(DF)
        
        # FUNCTION FOR TRIANGLE
        print('HERE_T')
        DF = add_triangle(smol_graph, model_labels, idx)
        print(DF.columns.to_list())
        if DF.shape[0] != 0:
            list_of_frames.append(DF)
    
    # Merge all feature dataframes together
    DF_ALL = reduce(lambda df1,df2: pd.merge(df1,df2,on='__id'), list_of_frames)
    print('DATAFRAME SHAPE: {}'.format(DF_ALL.shape))
    
    # Output final DF
    return DF_ALL

def add_pagerank(graph, model_labels, index):
    turicreate.config.set_runtime_config('TURI_DEFAULT_NUM_PYLAMBDA_WORKERS', 96)
    # If this particular graph is in the list of approved graphs, then continue, otherwise return empty dataframe
    if sgraph_idx[index] in feat_graph_map['pagerank']:
        # Create pagerank SFrame
        pr = pagerank.create(graph, verbose=False)
        pr_sframe = pr['pagerank']
        # Modifying output SFrame
        pr_df = pd.DataFrame(pr_sframe)
        pr_df = pr_df.drop('delta', axis=1)
        pr_df = pr_df[pr_df['__id'].isin(model_labels)].reset_index(drop=True)
        pr_df = pr_df.rename({'pagerank':'pr_{}'.format(index)}, axis=1)
        # Return modified dataframe
        return pr_df
    else:
        # Return empty dataframe
        return pd.DataFrame(columns=['__id'])

def add_weighted_pagerank(graph, model_labels, index):
    turicreate.config.set_runtime_config('TURI_DEFAULT_NUM_PYLAMBDA_WORKERS', 96)
    # If this particular graph is in the list of approved graphs, then continue, otherwise return empty dataframe
    if sgraph_idx[index] in feat_graph_map['pagerank_weight']:
        pr_w = pagerank_weighted(graph)
        pr_w_sframe = pr_w['__id', 'pagerank']
        # Modifying output SFrame
        pr_w_df = pd.DataFrame(pr_w_sframe)
        pr_w_df = pr_w_df[pr_w_df['__id'].isin(model_labels)].reset_index(drop=True)
        pr_w_df = pr_w_df.rename({'pagerank':'w_pr_{}'.format(index)}, axis=1)
        # Return modified dataframe
        return pr_w_df
    else:
        # Return empty dataframe
        return pd.DataFrame(columns=['__id'])   
      
def add_shortest_path(graph, model_labels, index, pagerank_dataframe, p1_companies_uuid):
    #mapping_for_pr = {1:1, 2:1}
    turicreate.config.set_runtime_config('TURI_DEFAULT_NUM_PYLAMBDA_WORKERS', 96)
    # If this particular graph is in the list of approved graphs, then continue, otherwise return empty dataframe
    if sgraph_idx[index] in feat_graph_map['shortest']:
        # Grab pagerank dataframe
        pr = pagerank_dataframe[['__id', 'pr_1']].sort_values(by='pr_1',ascending=False)
        pr = pr['__id'].to_list()
        # Find top 5 p1 companies 
        count = 0
        top_p1 = []
        while len(top_p1) < 5:
            if pr[count] in p1_companies_uuid:
                top_p1.append(pr[count])
            count += 1
            print(count)
        # Loop over top 5 companies to find shortest path to each
        list_of_frames = []
        for jdx,uuid in enumerate(top_p1):
            # Create shortest path SFrame
            sp = shortest_path.create(graph, source_vid=uuid, verbose=False)
            sp_sframe = sp['distance']
            # Modifying output SFrame
            sp_df = pd.DataFrame(sp_sframe)
            sp_df = sp_df[sp_df['__id'].isin(model_labels)].reset_index(drop=True)
            sp_df = sp_df.rename({'distance': 'spath_top_{}_{}'.format(index,jdx)}, axis=1)
            list_of_frames.append(sp_df)
        # Combine 5 shortest path columns
        sp_df = reduce(lambda df1,df2: pd.merge(df1,df2,on='__id'), list_of_frames)
        # Add minimum path (to top 5) column
        sp_df['spath_top_min_{}'.format(index)] = sp_df.min(axis=1)
        # Return modified dataframe
        return sp_df
    else:
        # Return empty dataframe
         return pd.DataFrame(columns=['__id'])

def add_weighted_shortest_path(graph, model_labels, index, pagerank_dataframe_weighted, p1_companies_uuid):
    #mapping_for_pr = {1:1, 2:1}
    turicreate.config.set_runtime_config('TURI_DEFAULT_NUM_PYLAMBDA_WORKERS', 96)
    # If this particular graph is in the list of approved graphs, then continue, otherwise return empty dataframe
    if sgraph_idx[index] in feat_graph_map['shortest_weight']:
        # Grab weighted pagerank dataframe
        pr = pagerank_dataframe_weighted[['__id', 'w_pr_1']].sort_values(by='w_pr_1',ascending=False)
        pr = pr['__id'].to_list()
        # Find top 5 p1 companies 
        count = 0
        top_p1 = []
        while len(top_p1) < 5:
            if pr[count] in p1_companies_uuid:
                top_p1.append(pr[count])
            count += 1
            print(count)
        # Loop over top 5 companies to find shortest path to each
        list_of_frames = []
        for jdx,uuid in enumerate(top_p1):
            # Create shortest path SFrame
            sp = shortest_path.create(graph, source_vid=uuid, weight_field='weight', verbose=False)
            sp_sframe = sp['distance']
            # Modifying output SFrame
            sp_df = pd.DataFrame(sp_sframe)
            sp_df = sp_df[sp_df['__id'].isin(model_labels)].reset_index(drop=True)
            sp_df = sp_df.rename({'distance': 'w_spath_top_{}_{}'.format(index,jdx)}, axis=1)
            list_of_frames.append(sp_df)
        # Combine 5 shortest path columns
        sp_df = reduce(lambda df1,df2: pd.merge(df1,df2,on='__id'), list_of_frames)
        # Add minimum path (to top 5) column
        sp_df['w_spath_top_min_{}'.format(index)] = sp_df.min(axis=1)
        # Return modified dataframe
        return sp_df
    else:
        # Return empty dataframe
         return pd.DataFrame(columns=['__id'])

def add_kcore(graph, model_labels, index):
    turicreate.config.set_runtime_config('TURI_DEFAULT_NUM_PYLAMBDA_WORKERS', 96)
    # If this particular graph is in the list of approved graphs, then continue, otherwise return empty dataframe
    if sgraph_idx[index] in feat_graph_map['kcore']:
        # Create kcore SFrame
        kc = kcore.create(graph, kmin=0, kmax=5, verbose=False)
        kc_sframe = kc['core_id'] 
        # Modifying output SFrame
        kc_df = pd.DataFrame(kc_sframe)
        kc_df = kc_df[kc_df['__id'].isin(model_labels)].reset_index(drop=True)
        kc_df = kc_df.rename({'core_id':'kc_{}'.format(index)}, axis=1)
        # Return modified dataframe
        return kc_df
    else:
        # Return empty dataframe
         return pd.DataFrame(columns=['__id'])

def add_degree(graph, model_labels, index):
    turicreate.config.set_runtime_config('TURI_DEFAULT_NUM_PYLAMBDA_WORKERS', 96)
    # If this particular graph is in the list of approved graphs, then continue, otherwise return empty dataframe
    if sgraph_idx[index] in feat_graph_map['degree']:
        # Create degree SGraph
        deg = degree_counting.create(graph)
        deg_sgraph = deg['graph'] 
        # Modifying output SFrame
        deg_df = pd.DataFrame(deg_sgraph.vertices[['__id', 'in_degree', 'out_degree']])
        deg_df = deg_df[deg_df['__id'].isin(model_labels)].reset_index(drop=True)
        deg_df = deg_df.rename({'in_degree':'in_deg_{}'.format(index),'out_degree':'out_deg_{}'.format(index)}, axis=1)
        # Return modified dataframe
        return deg_df
    else:
        # Return empty dataframe
         return pd.DataFrame(columns=['__id'])
        
def add_triangle(graph, model_labels, index):
    turicreate.config.set_runtime_config('TURI_DEFAULT_NUM_PYLAMBDA_WORKERS', 96)
    # If this particular graph is in the list of approved graphs, then continue, otherwise return empty dataframe
    if sgraph_idx[index] in feat_graph_map['triangle']:
        # Create triangle counting SFrame
        tc = triangle_counting.create(graph, verbose=False)
        tc_sframes = tc['triangle_count']
        # Modifying output SFrame
        tri_df = pd.DataFrame(tc_sframes)
        tri_df = tri_df[tri_df['__id'].isin(model_labels)].reset_index(drop=True)
        tri_df = tri_df.rename({'triangle_count':'tri_{}'.format(index)},axis=1)
        # Return modified dataframe
        return tri_df
    else:
        # Return empty dataframe
         return pd.DataFrame(columns=['__id'])
        
def update_pagerank_weight(src, edge, dst):
    if src['__id'] != dst['__id']: # ignore self-links
        dst['pagerank'] += src['prev_pagerank'] * edge['weight']
    return (src, edge, dst)

def update_pagerank_reset_prob(src, edge, dst):
    global reset
    if src['__id'] != dst['__id']: # ignore self-links
        dst['pagerank'] *= (1 - reset)
        dst['pagerank'] += reset
    return (src, edge, dst)

def update_pagerank_prev_to_current(src, edge, dst):
    if src['__id'] != dst['__id']: # ignore self-links
        src['prev_pagerank'] = src['pagerank']
    return (src, edge, dst)

def sum_weight(src, edge, dst):
    if src['__id'] != dst['__id']: # ignore self-links
        src['total_weight'] += edge['weight']
    return src, edge, dst

def make_pagerank_zero(src, edge, dst):
    if src['__id'] != dst['__id']: # ignore self-links
        dst['pagerank'] = 0
    return src, edge, dst

def update_l1_delta(src, edge, dst):
    if src['__id'] != dst['__id']: # ignore self-links
        dst['l1_delta'] = abs(dst['pagerank'] - dst['prev_pagerank'])
        src['l1_delta'] = abs(src['pagerank'] - src['prev_pagerank'])
    return src, edge, dst

def normalize_weight(src, edge, dst):
    if src['__id'] != dst['__id']: # ignore self-links
        edge['weight'] /= src['total_weight']
    return src, edge, dst

def pagerank_weighted(input_graph, reset_prob=0.15, threshold=0.01, max_iterations=3):
    g = SGraph(input_graph.vertices, input_graph.edges)
    global reset
    reset = reset_prob
    # compute normalized edge weight
    g.vertices['total_weight'] = 0.0
    g = g.triple_apply(sum_weight, ['total_weight'])
    g = g.triple_apply(normalize_weight, ['weight'])
    del g.vertices['total_weight']
    # initialize vertex field
    g.vertices['prev_pagerank'] = 1.0
    it = 0
    total_l1_delta = len(g.vertices)
    start = time.time()
    while(total_l1_delta > threshold and it < max_iterations):
        if 'pagerank' not in g.get_vertex_fields():
            g.vertices['pagerank'] = 0.0
        else:
            g = g.triple_apply(make_pagerank_zero, ['pagerank'])
        g = g.triple_apply(update_pagerank_weight, ['pagerank'])
        g = g.triple_apply(update_pagerank_reset_prob, ['pagerank'])
        if 'l1_delta' not in g.get_vertex_fields():
            g.vertices['l1_delta'] = (g.vertices['pagerank'] - g.vertices['prev_pagerank']).apply(lambda x: abs(x))
        else:
            g = g.triple_apply(update_l1_delta, ['l1_delta'])
        total_l1_delta = g.vertices['l1_delta'].sum()
        g = g.triple_apply(update_pagerank_prev_to_current, ['prev_pagerank'])
        print ("Iteration %d: total pagerank changed in L1 = %f" % (it, total_l1_delta))
        it = it + 1
    print ("Weighted pagerank finished in: %f secs" % (time.time() - start))
    del g.vertices['prev_pagerank']
    return g.vertices