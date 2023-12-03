VISION_SYSTEM_PROMPT = """<Principle>
You will evaluate the behavior of the person in the sequences of images.
They show discrete parts of the whole continuous behavior.
You should only evaluate the parts you can rate based on the given images.
Remember, you're evaluating the given parts to evaluate the whole continuous behavior, and you'll connect them later to evaluate the whole.
Never add your own judgment. Evaluate only in the contents of images themselves.

<Evaluate Format>
rubric_1: (original sentence of rubric)
score: Good / Fair / Poor / (Undetermined) - evaluate by 3 steps
reason: (Explain why did you rated it that way)

rubric_2: (original sentence of rubric)
..."""


AUDIO_SYSTEM_PROMPT = """<Principle>
You will evaluate the behavior of the person in the text.
You should only evaluate based on the given text.
Never add your own judgment. Evaluate only in the contents of text themselves.

<Evaluate Format>
rubric_1: (original sentence of rubric)
score: Good / Fair / Poor / (Undetermined) - evaluate by 3 steps
reason: (Explain why did you rated it that way)

rubric_2: (original sentence of rubric)
..."""


USER_PROMPT_TEMPLATE = """Evaluate the behavior's actions based on the <RUBRIC> provided

<RUBRIC>
{rubrics}"""


FINAL_EVALUATION_PROMPT = """
You see the following list of texts that evaluate forward roll:
{evals}
Write an full text that synthesizes and summarizes the contents of all the text above.
Each evaluates a specific part, and you should combine them based on what was evaluated in each part.
The way to combine them is 'OR', not 'AND', which means you only need to evaluate the parts by choosing best one not to average the whole thing.
Concatenate based on what was evaluated, if anything.

Example:
<Video Evaluation>

**RUBRIC_1: (original sentence of rubric)**
SCORE: Good / Fair / Poor / (Undetermined) - evaluate by 3 steps
REASON: (Explain why did you rated it that way)

**RUBRIC_2: (original sentence of rubric)**
....
-----END of Video-----

<Audio Evaluation>

**RUBRIC_1: (original sentence of rubric)**
SCORE: Good / Fair / Poor / (Undetermined) - evaluate by 3 steps
REASON: (Explain why did you rated it that way)

**RUBRIC_2: (original sentence of rubric)**
....
-----END of Audio-----

**Overall opinion**

**Total score** : 1~10 / 10
(Explain how that above 'Total score' was calculated)

Output:
"""
