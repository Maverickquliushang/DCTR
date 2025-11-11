sys_prompt = (
    "Use the given knowledge graph triplets to answer the question. "
    "List all possible answers, each prefixed by \"ans:\"."
)

cot_prompt = (
    "List each answer on a new line prefixed by \"ans:\"."
)

sys_prompt_gpt = (
    "Use the retrieved triplets to identify relevant facts for answering the question. "
    "List each selected triplet prefixed with \"evidence:\"."
)
cot_prompt_gpt = cot_prompt

sys_prompt_rm_rank = sys_prompt
cot_prompt_rm_rank = (
    "First eliminate incorrect answers. "
    "Then list remaining answers with \"ans:\" and rank the most confident first."
)

sys_prompt_simp = (
    "Use the knowledge graph triplets to answer the question."
)
cot_prompt_simp = (
    "List each answer on a new line, starting with \"ans:\"."
)

icl_sys_prompt = (
    "Use the retrieved triplets to answer the question. "
    "Return each answer prefixed with \"ans:\"."
)

icl_cot_prompt = (
    "Step-by-step reasoning: "
    "List the most likely answers using \"ans:\". "
    "If info is missing, return \"ans: not available\"."
)

icl_cot_prompt_post = (
    "List the most likely answers using \"ans:\". "
    "If info is missing, return \"ans: not available\". "
    "Step-by-step reasoning:"
)

icl_user_prompt = """Triplets:
(Darth Vader, film.character.portrayed_by, James Earl Jones)
(Star Wars: A New Hope, film.film.character, Darth Vader)
(Star Wars: A New Hope, film.film.initial_release_date, 1977)
(Star Wars: The Empire Strikes Back, film.film.character, Darth Vader)
(Star Wars: The Empire Strikes Back, film.film.initial_release_date, 1980)
(Star Wars: Return of the Jedi, film.film.character, Darth Vader)
(Star Wars: Return of the Jedi, film.film.initial_release_date, 1983)
(Darth Vader, film.character.appeared_in, Star Wars: A New Hope)
(Darth Vader, film.character.appeared_in, Star Wars: The Empire Strikes Back)
(Darth Vader, film.character.appeared_in, Star Wars: Return of the Jedi)
(James Earl Jones, film.actor.film, Star Wars: A New Hope)
(James Earl Jones, film.actor.film, Star Wars: The Empire Strikes Back)
(James Earl Jones, film.actor.film, Star Wars: Return of the Jedi)

Question:
In which years did the character portrayed by James Earl Jones appear in Star Wars films?"""

icl_ass_prompt = """To find the years the character voiced by James Earl Jones appeared in Star Wars movies, we look for the character he portrayed and find the release dates of movies where that character appeared.

James Earl Jones voiced Darth Vader. 
Darth Vader appeared in:
- Star Wars: A New Hope (1977)
- Star Wars: The Empire Strikes Back (1980)
- Star Wars: Return of the Jedi (1983)

So, the answers are:

ans: 1977 (Star Wars: A New Hope)
ans: 1980 (Star Wars: The Empire Strikes Back)
ans: 1983 (Star Wars: Return of the Jedi)"""

noevi_sys_prompt = (
    "Answer the question. "
    "List the answers using \"ans:\" format."
)

noevi_cot_prompt = (
    "Step-by-step reasoning: "
    "List the most likely answers using \"ans:\". "
    "If info is missing, return \"ans: not available\"."
)
