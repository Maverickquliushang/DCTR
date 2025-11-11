import networkx as nx
from collections import deque
import numpy as np
import os
import pickle
import torch
import torch.nn.functional as F

from tqdm import tqdm
import heapq


class RetrieverDataset:
    def __init__(
            self,
            config,
            split,
            skip_no_path=True
    ):
        dataset_name = config['dataset']['name']
        processed_dict_list = self._load_processed(dataset_name, split)

        triple_score_dict = self._get_triple_scores(
            dataset_name, split, processed_dict_list)

        emb_dict = self._load_emb(
            dataset_name, config['dataset']['text_encoder_name'], split)

        self._assembly(
            processed_dict_list, triple_score_dict, emb_dict, skip_no_path)

    def _load_processed(
            self,
            dataset_name,
            split
    ):
        processed_file = os.path.join(
            f'data_files/{dataset_name}/processed/{split}.pkl')
        with open(processed_file, 'rb') as f:
            return pickle.load(f)

    def _get_triple_scores(
            self,
            dataset_name,
            split,
            processed_dict_list
    ):
        save_dir = os.path.join('data_files', dataset_name, 'triple_scores')
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, f'{split}.pth')

        if os.path.exists(save_file):
            return torch.load(save_file)

        triple_score_dict = dict()
        for i in tqdm(range(len(processed_dict_list))):
            sample_i = processed_dict_list[i]
            sample_i_id = sample_i['id']
            triple_scores_i, max_path_length_i = self._extract_paths_and_score(
                sample_i)

            triple_score_dict[sample_i_id] = {
                'triple_scores': triple_scores_i,
                'max_path_length': max_path_length_i
            }

        torch.save(triple_score_dict, save_file)

        return triple_score_dict

    def _extract_paths_and_score(self, sample):
        nx_g = self._get_nx_g_with_weights(
            sample['h_id_list'],
            sample['r_id_list'],
            sample['t_id_list'],
            weights=[1.0] * len(sample['h_id_list'])
        )

        path_list_ = []
        for q_entity_id in sample['q_entity_id_list']:
            for a_entity_id in sample['a_entity_id_list']:
                # 1. Maximum Flow Retrieval to get Hm
                hm_graph = self._maximum_flow_subgraph(nx_g, q_entity_id, a_entity_id)

                if hm_graph:
                    # 2. Dual-Flux Path Extraction to get P_lower and P_upper
                    p_lower, p_upper = self._dual_flux_path_extraction(
                        hm_graph, q_entity_id, a_entity_id
                    )

                    # The final subgraph is the union of the two paths
                    if p_lower:
                        path_list_.append(p_lower)
                    if p_upper:
                        path_list_.append(p_upper)

        unique_paths_str = {tuple(p) for p in path_list_}
        path_list_ = [list(p) for p in unique_paths_str]

        if not path_list_:
            max_path_length = None
        else:
            max_path_length = 0
        path_list = []
        for path in path_list_:
            num_triples_path = len(path) - 1
            if num_triples_path > max_path_length:
                max_path_length = num_triples_path

            triples_path = []
            for i in range(num_triples_path):
                h_id_i = path[i]
                t_id_i = path[i + 1]

                triple_id = nx_g.get_edge_data(h_id_i, t_id_i)['triple_id']
                triples_path.append([triple_id])

            path_list.append(triples_path)

        num_triples = len(sample['h_id_list'])
        triple_scores = self._score_triples(path_list, num_triples)

        return triple_scores, max_path_length

    def _get_nx_g(self, h_id_list, r_id_list, t_id_list, weights):
        nx_g = nx.DiGraph()
        num_triples = len(h_id_list)
        for i in range(num_triples):
            h_i, r_i, t_i, w_i = h_id_list[i], r_id_list[i], t_id_list[i], weights[i]

            nx_g.add_edge(h_i, t_i, triple_id=i, relation_id=r_i, weight=w_i)
        return nx_g

    def _maximum_flow_subgraph(self, nx_g, q_entity_id, a_entity_id):
        try:
            flow_value, partition = nx.maximum_flow(nx_g, q_entity_id, a_entity_id, capacity='weight')
            reachable, non_reachable = partition

            hm_nodes = list(reachable)
            hm_graph = nx_g.subgraph(hm_nodes).copy()

            if flow_value == 0:
                return None

            return hm_graph

        except nx.NetworkXError:
            return None

    def _dual_flux_path_extraction(self, hm_graph, q_entity_id, a_entity_id, density_threshold=0.0):

        if hm_graph is None or not hm_graph.has_node(q_entity_id) or not hm_graph.has_node(a_entity_id):
            return [], []

        try:

            all_paths = list(nx.all_simple_paths(hm_graph, source=q_entity_id, target=a_entity_id))
        except nx.NodeNotFound:
            return [], []

        if not all_paths:
            return [], []

        dense_paths = []
        path_metrics = []

        for path in all_paths:
            if len(path) < 2: continue

            weights = [hm_graph[u][v].get('weight', 1.0) for u, v in zip(path[:-1], path[1:])]

            # Density D(P) - Eq. 3
            density = np.mean(weights)

            if density >= density_threshold:
                # Bottleneck Flux Φ(P) - Eq. 4
                bottleneck_flux = min(weights)
                # Total Information Ψ(P) - Eq. 4
                total_information = sum(weights)

                dense_paths.append(path)
                path_metrics.append({
                    'density': density,
                    'bottleneck': bottleneck_flux,
                    'total_info': total_information
                })

        if not dense_paths:
            return [], []

        # 3. Find P_lower (MaxMin-InfoPath) - Eq. 5
        # Prioritize maximizing the bottleneck flux, then total information.
        lower_bound_candidate = max(
            zip(dense_paths, path_metrics),
            key=lambda x: (x[1]['bottleneck'], x[1]['total_info'])
        )
        p_lower = lower_bound_candidate[0]

        # 4. Find P_upper (MaxDensity-InfoPath) - Eq. 6 & 7
        # Find the maximum density among all dense paths.
        max_density = max(m['density'] for m in path_metrics)

        # Isolate paths with the maximum density (P*)
        max_density_paths_and_metrics = [
            (p, m) for p, m in zip(dense_paths, path_metrics) if m['density'] == max_density
        ]

        # From P*, select the one with the highest total information.
        upper_bound_candidate = max(
            max_density_paths_and_metrics,
            key=lambda x: x[1]['total_info']
        )
        p_upper = upper_bound_candidate[0]

        return p_lower, p_upper
    def _score_triples(
            self,
            path_list,
            num_triples
    ):
        triple_scores = torch.zeros(num_triples)

        for path in path_list:
            for triple_id_list in path:
                triple_scores[triple_id_list] = 1.

        return triple_scores

    def _load_emb(
            self,
            dataset_name,
            text_encoder_name,
            split
    ):
        file_path = f'data_files/{dataset_name}/emb/{text_encoder_name}/{split}.pth'
        dict_file = torch.load(file_path)

        return dict_file

    def _assembly(
            self,
            processed_dict_list,
            triple_score_dict,
            emb_dict,
            skip_no_path,
    ):
        self.processed_dict_list = []

        num_relevant_triples = []
        num_skipped = 0
        for i in tqdm(range(len(processed_dict_list))):
            sample_i = processed_dict_list[i]
            sample_i_id = sample_i['id']
            assert sample_i_id in triple_score_dict

            triple_score_i = triple_score_dict[sample_i_id]['triple_scores']
            max_path_length_i = triple_score_dict[sample_i_id]['max_path_length']

            num_relevant_triples_i = len(triple_score_i.nonzero())
            num_relevant_triples.append(num_relevant_triples_i)

            sample_i['target_triple_probs'] = triple_score_i
            sample_i['max_path_length'] = max_path_length_i

            if skip_no_path and (max_path_length_i in [None, 0]):
                num_skipped += 1
                continue

            sample_i.update(emb_dict[sample_i_id])

            sample_i['a_entity'] = list(set(sample_i['a_entity']))
            sample_i['a_entity_id_list'] = list(set(sample_i['a_entity_id_list']))

            # PE for topic entities.
            num_entities_i = len(sample_i['text_entity_list']) + len(sample_i['non_text_entity_list'])
            topic_entity_mask = torch.zeros(num_entities_i)
            topic_entity_mask[sample_i['q_entity_id_list']] = 1.
            topic_entity_one_hot = F.one_hot(topic_entity_mask.long(), num_classes=2)
            sample_i['topic_entity_one_hot'] = topic_entity_one_hot.float()

            self.processed_dict_list.append(sample_i)

        median_num_relevant = int(np.median(num_relevant_triples))
        mean_num_relevant = int(np.mean(num_relevant_triples))
        max_num_relevant = int(np.max(num_relevant_triples))

        print(f'# skipped samples: {num_skipped}')
        print(
            f'# relevant triples | median: {median_num_relevant} | mean: {mean_num_relevant} | max: {max_num_relevant}')

    def __len__(self):
        return len(self.processed_dict_list)

    def __getitem__(self, i):
        return self.processed_dict_list[i]


def collate_retriever(data):
    sample = data[0]

    h_id_list = sample['h_id_list']
    h_id_tensor = torch.tensor(h_id_list)

    r_id_list = sample['r_id_list']
    r_id_tensor = torch.tensor(r_id_list)

    t_id_list = sample['t_id_list']
    t_id_tensor = torch.tensor(t_id_list)

    num_non_text_entities = len(sample['non_text_entity_list'])

    return h_id_tensor, r_id_tensor, t_id_tensor, sample['q_emb'], \
        sample['entity_embs'], num_non_text_entities, sample['relation_embs'], \
        sample['topic_entity_one_hot'], sample['target_triple_probs'], sample['a_entity_id_list']
