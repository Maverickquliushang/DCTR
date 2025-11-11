import os
import pickle
from tqdm import tqdm

from ..model.llm_scorer import VLLMDynamicTripleScorer


class EmbInferDataset:
    """
    Processes raw dataset samples to create a structured format for training.
    It uses a VLLMDynamicTripleScorer to generate initial weights for all triples
    in the context graph of each sample.
    """

    def __init__(
            self,
            raw_set,
            entity_identifiers,
            save_path,
            llm_model_path,  # Add LLM model path as an argument
            skip_no_topic=True,
            skip_no_ans=True
    ):

        print("Initializing VLLM Dynamic Triple Scorer...")
        # Initialize the scorer with the specified model path and vLLM arguments.
        self.scorer = VLLMDynamicTripleScorer(
            model_name=llm_model_path,
            tensor_parallel_size=1,  # Adjust based on your GPU setup
            gpu_memory_utilization=0.9,
            trust_remote_code=True  # Required for some models like Llama-3
        )
        print("Scorer initialized successfully.")

        self.processed_dict_list = self._process(
            raw_set,
            entity_identifiers,
            save_path
        )

        self.skip_no_topic = skip_no_topic
        self.skip_no_ans = skip_no_ans

        processed_dict_list = []
        num_skipped_topic = 0
        num_skipped_ans = 0
        for processed_dict_i in self.processed_dict_list:
            if (len(processed_dict_i['q_entity_id_list']) == 0) and skip_no_topic:
                num_skipped_topic += 1
                continue

            if (len(processed_dict_i['a_entity_id_list']) == 0) and skip_no_ans:
                num_skipped_ans += 1
                continue

            processed_dict_list.append(processed_dict_i)
        self.processed_dict_list = processed_dict_list

        print(f"Skipped {num_skipped_topic} samples with no topic entity.")
        print(f"Skipped {num_skipped_ans} samples with no answer entity.")
        print(f'# raw samples: {len(raw_set)} | # final processed samples: {len(self.processed_dict_list)}')

    def _process(
            self,
            raw_set,
            entity_identifiers,
            save_path
    ):
        if os.path.exists(save_path):
            print(f"Loading pre-processed data from {save_path}")
            with open(save_path, 'rb') as f:
                return pickle.load(f)

        processed_dict_list = []
        for sample_i in tqdm(raw_set, desc="Processing Raw Samples with LLM Scorer"):
            processed_dict_i = self._process_sample(sample_i, entity_identifiers)
            processed_dict_list.append(processed_dict_i)

        print(f"Saving processed data to {save_path}")
        with open(save_path, 'wb') as f:
            pickle.dump(processed_dict_list, f)

        return processed_dict_list

    def _process_sample(
            self,
            sample,
            entity_identifiers
    ):
        question = sample['question']
        triples = sample['graph']

        # --- Entity and Relation Mapping (same as your original code) ---
        all_entities = set()
        all_relations = set()
        for (h, r, t) in triples:
            all_entities.add(h)
            all_relations.add(r)
            all_entities.add(t)

        entity_list = sorted(list(all_entities))
        text_entity_list = [e for e in entity_list if e not in entity_identifiers]
        non_text_entity_list = [e for e in entity_list if e in entity_identifiers]

        entity2id = {entity: i for i, entity in enumerate(text_entity_list + non_text_entity_list)}

        relation_list = sorted(list(all_relations))
        rel2id = {rel: i for i, rel in enumerate(relation_list)}

        triples_list = [(h, r, t) for (h, r, t) in triples]
        h_id_list = [entity2id[h] for (h, _, _) in triples_list]
        r_id_list = [rel2id[r] for (_, r, _) in triples_list]
        t_id_list = [entity2id[t] for (_, _, t) in triples_list]

        current_path_p_cur = []

        batch_size = 32

        triples_score_list = []
        if triples_list:
            for i in range(0, len(triples_list), batch_size):
                batch_triples = triples_list[i:i + batch_size]

                scores = self.scorer.batch_score(
                    query=question,
                    current_path=current_path_p_cur,
                    candidate_triples=batch_triples
                )
                triples_score_list.extend(scores)

        q_entity_id_list = [entity2id[e] for e in sample.get('q_entity', []) if e in entity2id]
        a_entity_id_list = [entity2id[e] for e in sample.get('a_entity', []) if e in entity2id]

        processed_dict = {
            'id': sample['id'],
            'question': question,
            'q_entity': sample.get('q_entity', []),
            'q_entity_id_list': q_entity_id_list,
            'text_entity_list': text_entity_list,
            'non_text_entity_list': non_text_entity_list,
            'relation_list': relation_list,
            'h_id_list': h_id_list,
            'r_id_list': r_id_list,
            't_id_list': t_id_list,
            'a_entity': sample.get('a_entity', []),
            'a_entity_id_list': a_entity_id_list,
            'triples_list': triples_list,
            'triples_score_list': triples_score_list
        }
        return processed_dict

    def __len__(self):
        return len(self.processed_dict_list)

    def __getitem__(self, i):
        sample = self.processed_dict_list[i]

        id = sample['id']
        q_text = sample['question']
        text_entity_list = sample['text_entity_list']
        relation_list = sample['relation_list']

        return id, q_text, text_entity_list, relation_list
