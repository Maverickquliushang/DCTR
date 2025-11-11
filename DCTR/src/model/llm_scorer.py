import torch
import re
from typing import List, Tuple

try:
    from vllm import LLM, SamplingParams
except ImportError:
    raise ImportError("Please install vLLM to use this scorer: 'pip install vllm'")

SCORING_PROMPT_TEMPLATE = """You are an expert in knowledge graph reasoning. Your task is to evaluate the relevance of a "Candidate Triple" for answering a "Query", given the "Current Reasoning Path".

### Instructions:
1.  Analyze the Query, the Current Reasoning Path, and the Candidate Triple.
2.  The Current Reasoning Path shows the steps already taken to answer the query.
3.  The Candidate Triple is the next potential step.
4.  Determine how relevant and logical the Candidate Triple is as the *next step* in the path.
5.  Output a single floating-point relevance score between 0.0 (completely irrelevant) and 1.0 (highly relevant and logical).
6.  Your response MUST contain ONLY the numerical score and nothing else.

### Input:
- **Query**: "{query}"
- **Current Reasoning Path**:
{path_text}
- **Candidate Triple**: ({candidate_head}, {candidate_relation}, {candidate_tail})

### Relevance Score:
"""


class VLLMDynamicTripleScorer:


    def __init__(self, model_name: str, **vllm_kwargs):

        self.llm = LLM(model=model_name, **vllm_kwargs)

        self.sampling_params = SamplingParams(
            temperature=0.0,  #
            top_p=1.0,
            max_tokens=10,
            stop=["\n"]
        )

    def _format_path(self, path: List[Tuple[str, str, str]]) -> str:

        if not path:
            return "- The path is currently empty."
        return "\n".join([f"- ({h}, {r}, {t})" for h, r, t in path])

    def _parse_score(self, generated_text: str) -> float:

        try:
            match = re.search(r"(\d+\.\d+|\d+)", generated_text)
            if match:
                return float(match.group(1))
            else:
                print(f"Warning: Could not parse score from LLM output: '{generated_text}'")
                return 0.0  # Return a default low score on failure
        except (ValueError, IndexError):
            print(f"Warning: Exception while parsing score from: '{generated_text}'")
            return 0.0

    def score(self, query: str, current_path: List[Tuple[str, str, str]],
              candidate_triple: Tuple[str, str, str]) -> float:

        return self.batch_score(query, current_path, [candidate_triple])[0]

    def batch_score(self, query: str, current_path: List[Tuple[str, str, str]],
                    candidate_triples: List[Tuple[str, str, str]]) -> List[float]:
        path_text = self._format_path(current_path)
        prompts = [
            SCORING_PROMPT_TEMPLATE.format(
                query=query,
                path_text=path_text,
                candidate_head=h,
                candidate_relation=r,
                candidate_tail=t
            ) for h, r, t in candidate_triples
        ]

        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)

        scores = [self._parse_score(output.outputs[0].text) for output in outputs]

        return scores