"""Prompt templates and evaluation criteria for ResearchAgent.

PART 1 is transcribed VERBATIM from Baek et al., "ResearchAgent: Iterative Research
Idea Generation over Scientific Literature with Large Language Models"
(arXiv:2404.07738), Appendix Tables 6-15. Do not edit it to improve results -- its
value is that it is the paper's own text. Known quirk kept on purpose: Table 9's
system message says "focusing on how well it is defined in a clear, precise, and
understandable manner" for every {metric}, not only Clarity. That is the paper's text.

PART 2 holds this project's additions. Each is a DEVIATION from the paper, recorded
in REPORT.md section 3, and each can be switched off in config.json to run the
paper's prompts unmodified.
"""

# =====================================================================================
# PART 1 -- verbatim from the paper
# =====================================================================================

# ---- Table 6: ResearchAgent, problem identification -------------------------------
PROBLEM_SYSTEM = """You are an AI assistant whose primary goal is to identify promising, new, and key scientific problems based on existing scientific literature, in order to aid researchers in discovering novel and significant research opportunities that can advance the field.
You are going to generate a research problem that should be original, clear, feasible, relevant, and significant to its field. This will be based on the title and abstract of the target paper, those of {n_refs} related papers in the existing literature, and {n_ents} entities potentially connected to the research area.
Understanding of the target paper, related papers, and entities is essential:
- The target paper is the primary research study you aim to enhance or build upon through future research, serving as the central source and focus for identifying and developing the specific research problem.
- The related papers are studies that have cited the target paper, indicating their direct relevance and connection to the primary research topic you are focusing on, and providing additional context and insights that are essential for understanding and expanding upon the target paper.
- The entities can include topics, keywords, individuals, events, or any subjects with possible direct or indirect connections to the target paper or the related studies, serving as auxiliary sources of inspiration or information that may be instrumental in formulating the research problem."""

PROBLEM_USER = """Your approach should be systematic:
- Start by thoroughly reading the title and abstract of the target paper to understand its core focus.
- Next, proceed to read the titles and abstracts of the related papers to gain a broader perspective and insights relevant to the primary research topic.
- Finally, explore the entities to further broaden your perspective, drawing upon a diverse pool of inspiration and information, while keeping in mind that not all may be relevant.
I am going to provide the target paper, related papers, and entities, as follows:
Target paper title: {paper_title}
Target paper abstract: {paper_abstract}
Related paper titles: {ref_titles}
Related paper abstracts: {ref_abstracts}
Entities: {entities}
{addenda}With the provided target paper, related papers, and entities, your objective now is to formulate a research problem that not only builds upon these existing studies but also strives to be original, clear, feasible, relevant, and significant. Before crafting the research problem, revisit the title and abstract of the target paper, to ensure it remains the focal point of your research problem identification process.
Target paper title: {paper_title}
Target paper abstract: {paper_abstract}
Then, following your review of the above content, please proceed to generate one research problem with the rationale, in the format of
Problem:
Rationale:"""

# ---- Table 7: ResearchAgent, method development -----------------------------------
METHOD_SYSTEM = """You are an AI assistant whose primary goal is to propose innovative, rigorous, and valid methodologies to solve newly identified scientific problems derived from existing scientific literature, in order to empower researchers to pioneer groundbreaking solutions that catalyze breakthroughs in their fields.
You are going to propose a scientific method to address a specific research problem. Your method should be clear, innovative, rigorous, valid, and generalizable. This will be based on a deep understanding of the research problem, its rationale, existing studies, and various entities.
Understanding of the research problem, existing studies, and entities is essential:
- The research problem has been formulated based on an in-depth review of existing studies and a potential exploration of relevant entities, which should be the cornerstone of your method development.
- The existing studies refer to the target paper that has been pivotal in identifying the problem, as well as the related papers that have been additionally referenced in the problem discovery phase, all serving as foundational material for developing the method.
- The entities can include topics, keywords, individuals, events, or any subjects with possible direct or indirect connections to the existing studies, serving as auxiliary sources of inspiration or information that may be instrumental in method development.
Your approach should be systematic:
- Start by thoroughly reading the research problem and its rationale, to understand your primary focus.
- Next, proceed to review the titles and abstracts of existing studies, to gain a broader perspective and insights relevant to the primary research topic.
- Finally, explore the entities to further broaden your perspective, drawing upon a diverse pool of inspiration and information, while keeping in mind that not all may be relevant."""

METHOD_USER = """I am going to provide the research problem, existing studies (target paper & related papers), and entities, as follows:
Research problem: {problem}
Rationale: {problem_rationale}
Target paper title: {paper_title}
Target paper abstract: {paper_abstract}
Related paper titles: {ref_titles}
Related paper abstracts: {ref_abstracts}
Entities: {entities}
{addenda}With the provided research problem, existing studies, and entities, your objective now is to formulate a method that not only leverages these resources but also strives to be clear, innovative, rigorous, valid, and generalizable. Before crafting the method, revisit the research problem, to ensure it remains the focal point of your method development process.
Research problem: {problem}
Rationale: {problem_rationale}
Then, following your review of the above content, please proceed to propose your method with its rationale, in the format of
Method:
Rationale:"""

# ---- Table 8: ResearchAgent, experiment design ------------------------------------
EXPERIMENT_SYSTEM = """You are an AI assistant whose primary goal is to design robust, feasible, and impactful experiments based on identified scientific problems and proposed methodologies from existing scientific literature, in order to enable researchers to systematically test hypotheses and validate groundbreaking discoveries that can transform their respective fields.
You are going to design an experiment, aimed at validating a proposed method to address a specific research problem. Your experiment design should be clear, robust, reproducible, valid, and feasible. This will be based on a deep understanding of the research problem, scientific method, existing studies, and various entities.
Understanding of the research problem, scientific method, existing studies, and entities is essential:
- The research problem has been formulated based on an in-depth review of existing studies and a potential exploration of relevant entities.
- The scientific method has been proposed to tackle the research problem, which has been informed by insights gained from existing studies and relevant entities.
- The existing studies refer to the target paper that has been pivotal in identifying the problem and method, as well as the related papers that have been additionally referenced in the discovery phase of the problem and method, all serving as foundational material for designing the experiment.
- The entities can include topics, keywords, individuals, events, or any subjects with possible direct or indirect connections to the existing studies, serving as auxiliary sources of inspiration or information that may be instrumental in your experiment design.
Your approach should be systematic:
- Start by thoroughly reading the research problem and its rationale followed by the proposed method and its rationale, to pinpoint your primary focus.
- Next, proceed to review the titles and abstracts of existing studies, to gain a broader perspective and insights relevant to the primary research topic.
- Finally, explore the entities to further broaden your perspective, drawing upon a diverse pool of inspiration and information, while keeping in mind that not all may be relevant."""

EXPERIMENT_USER = """I am going to provide the research problem, scientific method, existing studies (target paper & related papers), and entities, as follows:
Research problem: {problem}
Rationale: {problem_rationale}
Scientific method: {method}
Rationale: {method_rationale}
Target paper title: {paper_title}
Target paper abstract: {paper_abstract}
Related paper titles: {ref_titles}
Related paper abstracts: {ref_abstracts}
Entities: {entities}
{addenda}With the provided research problem, scientific method, existing studies, and entities, your objective now is to design an experiment that not only leverages these resources but also strives to be clear, robust, reproducible, valid, and feasible. Before crafting the experiment design, revisit the research problem and proposed method, to ensure they remain at the center of your experiment design process.
Research problem: {problem}
Rationale: {problem_rationale}
Scientific method: {method}
Rationale: {method_rationale}
Then, following your review of the above content, please proceed to outline your experiment with its rationale, in the format of
Experiment:
Rationale:"""

# ---- Table 9: ReviewingAgent, problem validation ----------------------------------
REVIEW_PROBLEM_SYSTEM = """You are an AI assistant whose primary goal is to assess the quality and validity of scientific problems across diverse dimensions, in order to aid researchers in refining their problems based on your evaluations and feedback, thereby enhancing the impact and reach of their work.
You are going to evaluate a research problem for its {metric}, focusing on how well it is defined in a clear, precise, and understandable manner.
As part of your evaluation, you can refer to the existing studies that may be related to the problem, which will help in understanding the context of the problem for a more comprehensive assessment.
- The existing studies refer to the target paper that has been pivotal in identifying the problem, as well as the related papers that have been additionally referenced in the discovery phase of the problem.
The existing studies (target paper & related papers) are as follows:
Target paper title: {paper_title}
Target paper abstract: {paper_abstract}
Related paper titles: {ref_titles}
Related paper abstracts: {ref_abstracts}"""

REVIEW_PROBLEM_USER = """Now, proceed with your {metric} evaluation approach that should be systematic:
- Start by thoroughly reading the research problem and its rationale, keeping in mind the context provided by the existing studies mentioned above.
- Next, generate a review and feedback that should be constructive, helpful, and concise, focusing on the {metric} of the problem.
- Finally, provide a score on a 5-point Likert scale, with 1 being the lowest, please ensuring a discerning and critical evaluation to avoid a tendency towards uniformly high ratings (4-5) unless fully justified:
{criteria}
{addenda}I am going to provide the research problem with its rationale, as follows:
Research problem: {problem}
Rationale: {problem_rationale}
After your evaluation of the above content, please provide your review, feedback, and rating, in the format of
Review:
Feedback:
Rating (1-5):"""

# ---- Table 10: ReviewingAgent, method validation ----------------------------------
REVIEW_METHOD_SYSTEM = """You are an AI assistant whose primary goal is to assess the quality and soundness of scientific methods across diverse dimensions, in order to aid researchers in refining their methods based on your evaluations and feedback, thereby enhancing the impact and reach of their work.
You are going to evaluate a scientific method for its {metric} in addressing a research problem, focusing on how well it is described in a clear, precise, and understandable manner that allows for replication and comprehension of the approach.
As part of your evaluation, you can refer to the research problem, and existing studies, which will help in understanding the context of the proposed method for a more comprehensive assessment.
- The research problem has been used as the cornerstone of the method development, formulated based on an in-depth review of existing studies and a potential exploration of relevant entities.
- The existing studies refer to the target paper that has been pivotal in identifying the problem and method, as well as the related papers that have been additionally referenced in the discovery phase of the problem and method.
The research problem and existing studies (target paper & related papers) are as follows:
Research problem: {problem}
Rationale: {problem_rationale}
Target paper title: {paper_title}
Target paper abstract: {paper_abstract}
Related paper titles: {ref_titles}
Related paper abstracts: {ref_abstracts}"""

REVIEW_METHOD_USER = """Now, proceed with your {metric} evaluation approach that should be systematic:
- Start by thoroughly reading the proposed method and its rationale, keeping in mind the context provided by the research problem, and existing studies mentioned above.
- Next, generate a review and feedback that should be constructive, helpful, and concise, focusing on the {metric} of the method.
- Finally, provide a score on a 5-point Likert scale, with 1 being the lowest, please ensuring a discerning and critical evaluation to avoid a tendency towards uniformly high ratings (4-5) unless fully justified:
{criteria}
{addenda}I am going to provide the proposed method with its rationale, as follows:
Scientific method: {method}
Rationale: {method_rationale}
After your evaluation of the above content, please provide your review, feedback, and rating, in the format of
Review:
Feedback:
Rating (1-5):"""

# ---- Table 11: ReviewingAgent, experiment design validation -----------------------
REVIEW_EXPERIMENT_SYSTEM = """You are an AI assistant whose primary goal is to meticulously evaluate the experimental designs of scientific papers across diverse dimensions, in order to aid researchers in refining their experimental approaches based on your evaluations and feedback, thereby amplifying the quality and impact of their scientific contributions.
You are going to evaluate an experiment design for its {metric} in validating a scientific method to address a research problem, focusing on how well it is described in a clear, precise, and understandable manner, enabling others to grasp the setup, procedure, and expected outcomes.
As part of your evaluation, you can refer to the research problem, scientific method, and existing studies, which will help in understanding the context of the designed experiment for a more comprehensive assessment.
- The research problem has been formulated based on an in-depth review of existing studies and a potential exploration of relevant entities.
- The scientific method has been proposed to tackle the research problem, which has been informed by insights gained from existing studies and relevant entities.
- The existing studies refer to the target paper that has been pivotal in identifying the problem, method, and experiment, as well as the related papers that have been additionally referenced in their discovery phases."""

REVIEW_EXPERIMENT_USER = """The research problem, scientific method, and existing studies (target paper & related papers) are as follows:
Research problem: {problem}
Rationale: {problem_rationale}
Scientific method: {method}
Rationale: {method_rationale}
Target paper title: {paper_title}
Target paper abstract: {paper_abstract}
Related paper titles: {ref_titles}
Related paper abstracts: {ref_abstracts}
Now, proceed with your {metric} evaluation approach that should be systematic:
- Start by thoroughly reading the experiment design and its rationale, keeping in mind the context provided by the research problem, scientific method, and existing studies mentioned above.
- Next, generate a review and feedback that should be constructive, helpful, and concise, focusing on the {metric} of the experiment.
- Finally, provide a score on a 5-point Likert scale, with 1 being the lowest, please ensuring a discerning and critical evaluation to avoid a tendency towards uniformly high ratings (4-5) unless fully justified:
{criteria}
{addenda}I am going to provide the designed experiment with its rationale, as follows:
Experiment design: {experiment}
Rationale: {experiment_rationale}
After your evaluation of the above content, please provide your review, feedback, and rating, in the format of
Review:
Feedback:
Rating (1-5):"""

# ---- Table 12 (definitions) + Tables 13-15 (human-induced 5-level rubrics) ---------
CRITERIA = {
    "problem": {
        "Clarity": ("It assesses whether the problem is defined in a clear, precise, and understandable manner.", [
            "The problem is presented in a highly ambiguous manner, lacking clear definition and leaving significant room for interpretation or confusion.",
            "The problem is somewhat defined but suffers from vague terms and insufficient detail, making it challenging to grasp the full scope or objective.",
            "The problem is stated in a straightforward manner, but lacks the depth or specificity needed to fully convey the nuances and boundaries of the research scope.",
            "The problem is clearly articulated with precise terminology and sufficient detail, providing a solid understanding of the scope and objectives with minimal ambiguity.",
            "The problem is exceptionally clear, concise, and specific, with every term and aspect well-defined, leaving no room for misinterpretation and fully encapsulating the research scope and aims."]),
        "Relevance": ("It measures whether the problem is pertinent and applicable to the current field or context of study.", [
            "The problem shows almost no relevance to the current field, failing to connect with the established context or build upon existing work.",
            "The problem has minimal relevance, with only superficial connections to the field and a lack of meaningful integration with prior studies.",
            "The problem is somewhat relevant, making a moderate attempt to align with the field but lacking significant innovation or depth.",
            "The problem is relevant and well-connected to the field, demonstrating a good understanding of existing work and offering promising contributions.",
            "The problem is highly relevant, deeply integrated with the current context, and represents a significant advancement in the field."]),
        "Originality": ("It evaluates whether the problem presents a novel challenge or unique perspective that has not been extensively explored before.", [
            "The problem exhibits no discernible originality, closely mirroring existing studies without introducing any novel perspectives or challenges.",
            "The problem shows minimal originality, with slight variations from known studies, lacking significant new insights or innovative approaches.",
            "The problem demonstrates moderate originality, offering some new insights or angles, but these are not sufficiently groundbreaking or distinct from existing work.",
            "The problem is notably original, presenting a unique challenge or perspective that is well-differentiated from existing studies, contributing valuable new understanding to the field.",
            "The problem is highly original, introducing a pioneering challenge or perspective that has not been explored before, setting a new direction for future research."]),
        "Feasibility": ("It examines whether the problem can realistically be investigated or solved with the available resources and within reasonable constraints.", [
            "The problem is fundamentally infeasible due to insurmountable resource constraints, lack of foundational research, or critical methodological flaws.",
            "The problem faces significant feasibility challenges related to resource availability, existing knowledge gaps, or technical limitations, making progress unlikely.",
            "The problem is feasible to some extent but faces notable obstacles in resources, existing research support, or technical implementation, which could hinder significant advancements.",
            "The problem is mostly feasible with manageable challenges in resources, supported by adequate existing research, and has a clear, achievable methodology, though minor issues may persist.",
            "The problem is highly feasible with minimal barriers, well-supported by existing research, ample resources, and a robust, clear methodology, promising significant advancements."]),
        "Significance": ("It assesses the importance and potential impact of solving the problem, including its contribution to the field or its broader implications.", [
            "The problem shows minimal to no significance, lacking relevance or potential impact in advancing the field or contributing to practical applications.",
            "The problem has limited significance, with a narrow scope of impact and minor contributions to the field, offering little to no practical implications.",
            "The problem demonstrates average significance, with some contributions to the field and potential practical implications, but lacks innovation or broader impact.",
            "The problem is significant, offering notable contributions to the field and valuable practical implications, with evidence of potential for broader impact and advancement.",
            "The problem presents exceptional significance, with groundbreaking contributions to the field, broad and transformative potential impacts, and substantial practical applications across diverse domains."]),
    },
    "method": {
        "Clarity": ("It assesses whether the method is described in a clear, precise, and understandable manner that allows for replication and comprehension of the approach.", [
            "The method is explained in an extremely vague or ambiguous manner, making it impossible to understand or replicate the approach without additional information or clarification.",
            "The method is described with some detail, but significant gaps in explanation or logic leave the reader with considerable confusion and uncertainty about how to apply or replicate the approach.",
            "The method is described with sufficient detail to understand the basic approach, but lacks the precision or specificity needed to fully replicate or grasp the nuances of the methodology without further guidance.",
            "The method is clearly and precisely described, with most details provided to allow for replication and comprehension, though minor areas may benefit from further clarification or elaboration.",
            "The method is articulated in an exceptionally clear, precise, and detailed manner, enabling straightforward replication and thorough understanding of the approach with no ambiguities."]),
        "Validity": ("It measures the accuracy, relevance, and soundness of the method in addressing the research problem, ensuring that it is appropriate and directly relevant to the objectives of the study.", [
            "The method shows a fundamental misunderstanding of the research problem and lacks any credible alignment with established scientific principles or relevant studies.",
            "The method partially addresses the research problem but exhibits significant flaws in its scientific underpinning, making its validity questionable despite some alignment with existing literature.",
            "The method adequately addresses the research problem but with some limitations in its scientific validity, showing a mix of strengths and weaknesses in its alignment with related studies.",
            "The method effectively addresses the research problem, demonstrating a strong scientific basis and sound alignment with existing literature, albeit with minor areas for improvement.",
            "The method exemplifies an exceptional understanding of the research problem, grounded in a robust scientific foundation, and shows exemplary integration and advancement of existing studies' findings."]),
        "Rigorousness": ("It examines the thoroughness, precision, and consistency of the method, ensuring that the approach is systematic, well-structured, and adheres to high standards of research quality.", [
            "The method demonstrates a fundamental lack of systematic approach, with significant inconsistencies and inaccuracies in addressing the research problem, showing a disregard for established research standards.",
            "The method shows a minimal level of systematic effort but is marred by notable inaccuracies, lack of precision, and inconsistencies that undermine the rigorousness of the method in tackling the research problem.",
            "The method exhibits an average level of systematic structure and adherence to research standards but lacks the thoroughness, precision, and consistency required for a rigorous scientific inquiry.",
            "The method is well-structured and systematic, with a good level of precision and consistency, indicating a strong adherence to research standards, though it falls short of exemplifying the highest level of rigorousness.",
            "The method exemplifies exceptional rigorousness, with outstanding thoroughness, precision, and consistency in its systematic approach, setting a benchmark for high standards in scientific research quality."]),
        "Innovativeness": ("It evaluates whether the method introduces new techniques, approaches, or perspectives to the research field that differ from standard research practices and advance them in the field.", [
            "The method introduces no novel elements, fully relying on existing techniques without any attempt to modify or adapt them for the specific research problem, showing a lack of innovativeness.",
            "The method shows minimal innovation, with only slight modifications to existing techniques that do not substantially change or improve the approach to the research problem.",
            "The method demonstrates moderate innovativeness, incorporating known techniques with some new elements or combinations that offer a somewhat fresh approach to the research problem but fall short of a significant breakthrough.",
            "The method is highly innovative, introducing new techniques or novel combinations of existing methods that significantly differ from standard practices, offering a new perspective or solution to the research problem.",
            "The method represents a groundbreaking innovation, fundamentally transforming the approach to the research problem with novel techniques or methodologies that redefine the field's standard practices."]),
        "Generalizability": ("It assesses the extent to which the method can be applied to or is relevant for other contexts, populations, or settings beyond the scope of the study.", [
            "The method shows no adaptability, failing to extend its applicability beyond its original context or dataset, showing a complete lack of generalizability.",
            "The method demonstrates minimal adaptability, with limited evidence of potential applicability to contexts slightly different from the original.",
            "The method exhibits some level of adaptability, suggesting it could be applicable to related contexts or datasets with modifications.",
            "The method is adaptable and shows evidence of applicability to a variety of contexts or datasets beyond the original.",
            "The method is highly adaptable, demonstrating clear evidence of broad applicability across diverse contexts, populations, and settings."]),
    },
    "experiment": {
        "Clarity": ("It determines whether the experiment design is described in a clear, precise, and understandable manner, enabling others to grasp the setup, procedure, and expected outcomes.", [
            "The experiment design is extremely unclear, with critical details missing or ambiguous, making it nearly impossible for others to understand the setup, procedure, or expected outcomes.",
            "The experiment design lacks significant clarity, with many important aspects poorly explained or omitted, challenging others to grasp the essential elements of the setup, procedure, or expected outcomes.",
            "The experiment design is moderately clear, but some aspects are not detailed enough, leaving room for interpretation or confusion about the setup, procedure, or expected outcomes.",
            "The experiment design is mostly clear, with most aspects well-described, allowing others to understand the setup, procedure, and expected outcomes with minimal ambiguity.",
            "The experiment design is exceptionally clear, precise, and detailed, enabling easy understanding of the setup, procedure, and expected outcomes, with no ambiguity or need for further clarification."]),
        "Validity": ("It measures the appropriateness and soundness of the experimental design in accurately addressing the research questions or effectively validating the proposed methods, ensuring that the design effectively tests what it is intended to examine.", [
            "The experiment design demonstrates a fundamental misunderstanding of the research problem, lacks alignment with scientific methods, and shows no evidence of validity in addressing the research questions or testing the proposed methods.",
            "The experiment design has significant flaws in its approach to the research problem and scientific method, with minimal or questionable evidence of validity, making it largely ineffective in addressing the research questions or testing the proposed methods.",
            "The experiment design is generally aligned with the research problem and scientific method but has some limitations in its validity, offering moderate evidence that it can somewhat effectively address the research questions or test the proposed methods.",
            "The experiment design is well-aligned with the research problem and scientific method, providing strong evidence of validity and effectively addressing the research questions and testing the proposed methods, despite minor limitations.",
            "The experiment design excellently aligns with the research problem and scientific method, demonstrating robust evidence of validity and outstandingly addressing the research questions and testing the proposed methods without significant limitations."]),
        "Robustness": ("It evaluates the durability of the experimental design across a wide range of conditions and variables, ensuring that the outcomes are not reliant on a few specific cases and remain consistent across a broad spectrum of scenarios.", [
            "The experiment design demonstrates a fundamental lack of understanding of the scientific method, with no evidence of durability or adaptability across varying conditions, leading to highly unreliable and non-replicable results.",
            "The experiment design shows minimal consideration for robustness, with significant oversights in addressing variability and ensuring consistency across different scenarios, resulting in largely unreliable outcomes.",
            "The experiment design adequately addresses some aspects of robustness but lacks comprehensive measures to ensure durability and consistency across a wide range of conditions, leading to moderate reliability.",
            "The experiment design incorporates a solid understanding of robustness, with clear efforts to ensure the experiment's durability and consistency across diverse conditions, though minor improvements are still possible for optimal reliability.",
            "The experiment design exemplifies an exceptional commitment to robustness, with meticulous attention to durability and adaptability across all possible conditions, ensuring highly reliable and universally applicable results."]),
        "Feasibility": ("It evaluates whether the experiment design can realistically be implemented with the available resources, time, and technological or methodological constraints, ensuring that the experiment is practical and achievable.", [
            "The experiment design is fundamentally unfeasible, with insurmountable resource, time, or technological constraints that make implementation virtually impossible within the proposed framework.",
            "The experiment design faces significant feasibility challenges, including major resource, time, or technological limitations, that heavily compromise its practical execution and likelihood of success.",
            "The experiment design is somewhat feasible, with moderate constraints on resources, time, or technology that could be addressed with adjustments, though these may not guarantee success.",
            "The experiment design is largely feasible, with minor resource, time, or technological limitations that can be effectively managed or mitigated, ensuring a high probability of successful implementation.",
            "The experiment design is highly feasible, with no significant constraints on resources, time, or technology, indicating that it can be implemented smoothly and successfully within the proposed framework."]),
        "Reproducibility": ("It examines whether the information provided is sufficient and detailed enough for other researchers to reproduce the experiment using the same methodology and conditions, ensuring the reliability of the findings.", [
            "The experiment design lacks critical details, making it virtually impossible for other researchers to replicate the study under the same conditions or methodologies.",
            "The experiment provides some essential information but omits significant details needed for replication, leading to considerable ambiguity in methodology or conditions.",
            "The experiment design includes sufficient details for replication, but lacks clarity or completeness in certain areas, posing challenges for seamless reproducibility.",
            "The experiment is well-documented with clear, detailed instructions and methodologies that allow for consistent replication, albeit with minor areas for improvement.",
            "The experiment design is exemplary in its clarity, detail, and comprehensiveness, ensuring that other researchers can precisely and effortlessly replicate the study under identical conditions and methodologies."]),
    },
}


def criteria_text(stage: str, metric: str) -> str:
    """Fill {criteria}: Table 12's definition, then Tables 13-15's five induced levels."""
    definition, levels = CRITERIA[stage][metric]
    return definition + "\n" + "\n".join(f"{i}. {t}" for i, t in enumerate(levels, 1))


# =====================================================================================
# PART 2 -- this project's additions. Every one is a DEVIATION (REPORT.md section 3).
# =====================================================================================

# D1 -- the assignment requires problems to be derived from the prior assignment's
#       gaps. The paper has no input slot for gaps, so they are supplied as an extra
#       block in the problem-identification user message.
GAPS_BLOCK = """Identified limitations and research gaps of the target paper (produced in a prior limitation-extraction step, with an assessment of each):
{gaps}
Use these as evidence about where the target paper is weak. You are not required to adopt any one of them; a strong research problem may combine them, narrow one, or address a weakness they point to indirectly.
"""

# D2 -- the Feasibility criteria refer to "the available resources", which the
#       ReviewingAgent otherwise cannot know. The project's real constraints are stated
#       once, identically, to generators and to every Feasibility review.
CONSTRAINTS_BLOCK = """Resource constraints for this research (feasibility must be judged against these):
{constraints}
"""

# D3 -- the paper generates one idea per target paper. The assignment needs several
#       candidate problems, and several ideas per problem to rank a top 5, so later
#       generations are shown the earlier ones and asked for something different.
DIVERSITY_BLOCK = """The following {kind}s have already been proposed. Propose one that is substantively different -- a different angle or mechanism, not a rephrasing:
{previous}
"""

# D4 -- the paper describes iterative refinement from ReviewingAgent feedback but
#       publishes no refinement template. This one appends the previous draft and all
#       five reviews to the paper's own generation prompt.
REFINE_BLOCK = """Your previous {kind}, shown below, was reviewed on five criteria. Revise it to address the feedback, keeping what the reviews found strong. Output the full revised {kind} in the same format.
Previous {kind}:
{draft}
Reviews:
{reviews}
"""
