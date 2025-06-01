def direct_prompting(question, options):
    """
    Directly prompts the user with a question and options.
    """
    prompt = f"Question: {question}\n" \
                f"Options: {options}\n" \
                f"You are a legal expert, please respond only with the selected option like Yes/No, using the following format: '''Option: [selected option]'''"
    return prompt

def cot_prompting(question, options):
    """
    Chain of Thought prompting for legal questions.
    """

    format = f"Reasoning: [step-by-step thoughts]. " \
            f"Answer: [selected option (like Yes or No)]\n"
    prompt = f"{question}\n" \
                f"Options: {options}\n" \
                "Please think step by step and provide your reasoning to make sure we have the right answer.\n" \
                f"Your response should be in the following format: '''{format}'''"
    return prompt

# Số lượng subfields để phân loại câu hỏi luật (Question Domains)
NUM_LD = 5
# Số lượng subfields để phân loại các đáp án luật (Options Domains)
NUM_LOD = 2

# 1. Phân loại câu hỏi vào các subfields của luật
def get_question_domains_prompt_legal(question):
    """
    Trả về:
      - question_classifier: vai trò của mô hình (system role) khi phân loại câu hỏi luật.
      - prompt_get_question_domain: prompt chi tiết để yêu cầu LLM chọn NUM_LD subfields.
    """
    # Định dạng mẫu output: "Legal Field: Field0 | Field1 | Field2 | Field3 | Field4"
    question_domain_format = "Legal Field: " + " | ".join([f"Field{i}" for i in range(NUM_LD)])
    
    question_classifier = (
        "You are a legal expert who specializes in categorizing a specific legal scenario "
        "into precise subfields of law."
    )
    prompt_get_question_domain = (
        f"You need to complete the following steps:\n"
        f"1. Carefully read the legal scenario described in the question: '''{question}'''.\n"
        f"2. Based on the legal context, classify this question into {NUM_LD} different subfields of law.\n"
        f"3. You should output in exactly the same format as '{question_domain_format}'."
    )
    return question_classifier, prompt_get_question_domain


# 2. Phân tích câu hỏi từ góc nhìn từng subfield (legal)
def get_question_analysis_prompt_legal(question, question_domain):
    """
    Trả về:
      - question_analyzer: vai trò của mô hình (system role) đã giả lập là chuyên gia pháp lý trong domain cụ thể.
      - prompt_get_question_analysis: prompt chi tiết để yêu cầu phân tích tình huống pháp lý.
    """
    question_analyzer = (
        f"You are a legal expert in the domain of {question_domain}. "
        f"Drawing upon your specialized legal knowledge, you will examine the facts, issues, "
        f"and relevant legal principles presented in this scenario."
    )
    prompt_get_question_analysis = (
        f"Please meticulously examine the legal scenario outlined in this question: '''{question}'''. "
        f"From your area of specialization ({question_domain}), identify the key legal issues, "
        f"relevant statutes or precedents, and any potential risks or ambiguities. "
        f"Highlight what you consider to be the central legal question and any sub‐issues that require attention."
    )
    return question_analyzer, prompt_get_question_analysis


# 3. Phân loại các option vào subfields của luật
def get_options_domains_prompt_legal(question, options):
    """
    Trả về:
      - options_classifier: vai trò của mô hình (system role) khi phân loại các đáp án vào domain.
      - prompt_get_options_domain: prompt chi tiết để yêu cầu phân loại options vào NUM_LOD subfields luật.
    """
    options_domain_format = "Legal Field: " + " | ".join([f"Field{i}" for i in range(NUM_LOD)])
    
    options_classifier = (
        "As a legal expert, you possess the ability to discern the two most relevant subfields "
        "of law needed to evaluate the multiple‐choice answers for this legal scenario."
    )
    prompt_get_options_domain = (
        f"You need to complete the following steps:\n"
        f"1. Carefully read the legal scenario presented in the question: '''{question}'''.\n"
        f"2. The possible answers (options) are: '''{options}'''. "
        f"   Understand how each option relates to the question.\n"
        f"3. Your core objective is to categorize these options into {NUM_LOD} distinct subfields of law. "
        f"You should output in exactly the same format as '{options_domain_format}'."
    )
    return options_classifier, prompt_get_options_domain


# 4. Phân tích từng option từ góc nhìn của một domain luật
def get_options_analysis_prompt_legal(question, options, op_domain, question_analysis):
    """
    Trả về:
      - option_analyzer: vai trò của mô hình (system role) đã giả lập là chuyên gia pháp lý trong op_domain.
      - prompt_get_options_analyses: prompt chi tiết để yêu cầu phân tích từng option dựa trên
        cả question_analysis (5 domain) và op_domain (1 trong 2 domain liên quan).
    """
    option_analyzer = (
        f"You are a legal expert specialized in the {op_domain} domain. "
        f"You are adept at analyzing multiple‐choice questions in legal exams and assessing the validity "
        f"of each answer choice based on legal reasoning and precedents."
    )

    # Xây dựng prompt với context là 5 analysis của question
    prompt_get_options_analyses = (
        f"Regarding the question: '''{question}''', we have obtained detailed analyses "
        f"from experts in five distinct legal subfields. \n"
    )
    for _domain, _analysis in question_analysis.items():
        prompt_get_options_analyses += (
            f"The analysis from a {_domain} expert suggests: {_analysis}\n"
        )

    prompt_get_options_analyses += (
        f"The following are the answer choices: '''{options}'''.\n"
        f"Based on the team’s multi‐domain insights above, you are required to:\n"
        f"  • Understand how each option aligns or conflicts with the legal issues identified.\n"
        f"  • Scrutinize each option individually from your perspective in {op_domain}, "
        f"    and evaluate whether it is legally sound or must be eliminated.\n"
        f"  • Pay close attention to subtle distinctions: some options may appear correct at first glance "
        f"    but could be misleading or inconsistent with relevant statutes or precedents."
    )
    return option_analyzer, prompt_get_options_analyses


# 5. Tạo prompt cuối cùng khi đã có question_analyses (5) và option_analyses (2)
def get_final_answer_prompt_analonly_legal(question, options, question_analyses, option_analyses):
    """
    Trả về prompt cuối cùng yêu cầu LLM tổng hợp từ:
      - question (chuỗi)
      - options (chuỗi)
      - question_analyses (dict: 5 phân tích theo domain)
      - option_analyses (dict: 2 phân tích theo domain cho options)
    và chọn đáp án đúng (chỉ trả về chữ cái A/B/C/D/E).
    """
    prompt = (
        f"Question: {question}\n"
        f"Options: {options}\n"
        f"Answer: Let's work this out in a step‐by‐step way to be sure we arrive at the correct legal answer.\n"
        f"Step 1: Decode the question. We have a team of experts from five legal subfields who analyzed it. \n"
    )
    for _domain, _analysis in question_analyses.items():
        prompt += f"Insight from an expert in {_domain} suggests: {_analysis}\n"

    prompt += (
        f"Step 2: Evaluate each answer choice individually, based on both the specifics of the legal scenario "
        f"and your legal expertise. Pay close attention to distinctions among the options. "
        f"Some options may seem plausible but conflict with statutes, regulations, or case law. \n"
        f"We also have in‐depth assessments from two specialized domains for the options: \n"
    )
    for _domain, _analysis in option_analyses.items():
        prompt += f"Assessment from a {_domain} expert for the options suggests: {_analysis}\n"

    prompt += (
        f"Step 3: Based on the above insights, select the optimal choice to answer the question.\n"
        f"Points to note:\n"
        f"  1. Các phân tích được cung cấp phải dẫn bạn tới đáp án hợp lý về mặt pháp lý.\n"
        f"  2. Bất kỳ lựa chọn nào mâu thuẫn với nguyên tắc pháp luật, đạo luật, hoặc án lệ quan trọng đều không thể là đáp án đúng.\n"
        f"  3. Vui lòng trả lời CHỈ với chữ cái của lựa chọn đúng (A, B, C, D hoặc E), theo format: 'Option: [Chữ cái]'.\n"
        f"     Chúng tôi cần đúng chữ cái, không cần trích dẫn nội dung đầy đủ của đáp án."
    )
    return prompt


# 6. Tạo prompt cuối cùng khi đã có báo cáo tổng hợp (synthesized report)
def get_final_answer_prompt_wsyn_legal(syn_report):
    """
    Trả về prompt cuối cùng chỉ cần dựa vào syn_report (báo cáo đã tổng hợp từ 
    các question_analyses và option_analyses) để chọn đáp án.
    """
    prompt = (
        f"Here is a synthesized legal reasoning report:\n{syn_report}\n"
        f"Based on the above report, select the optimal choice to answer the question.\n"
        f"Points to note:\n"
        f"  1. Các phân tích có trong báo cáo phải dẫn bạn tới đáp án hợp lý về mặt pháp lý.\n"
        f"  2. Bất kỳ lựa chọn nào mâu thuẫn với nguyên tắc pháp luật, quy định, hoặc án lệ quan trọng đều không thể là đáp án đúng.\n"
        f"  3. Vui lòng trả lời CHỈ với chữ cái của lựa chọn đúng (A, B, C, D hoặc E), theo format: 'Option: [Chữ cái]'.\n"
        f"     Chúng tôi cần đúng chữ cái, không cần trích dẫn nội dung đầy đủ của đáp án."
    )
    return prompt

def get_synthesized_report_prompt_legal(question_analyses, option_analyses):
    """
    Trả về:
      - synthesizer: Vai trò của mô hình (system role) khi tổng hợp các phân tích pháp lý.
      - prompt: Câu lệnh chi tiết để LLM tổng hợp từ các báo cáo của nhiều chuyên gia pháp lý.
    """
    synthesizer = (
        "You are a legal decision maker who excels at summarizing and synthesizing "
        "insights from multiple domain experts in law."
    )

    syn_report_format = (
        "Key Legal Principles: [extracted key legal principles]\n"
        "Total Analysis: [synthesized comprehensive analysis]\n"
    )

    prompt = (
        "Here are several reports from different legal domain experts:\n\n"
        f"{question_analyses}\n"
        f"{option_analyses}\n\n"
        "You need to complete the following steps:\n"
        "1. Carefully review all of the following expert reports.\n"
        "2. Extract key legal principles and precedents mentioned.\n"
        "3. Derive a comprehensive, summarized analysis that integrates these principles.\n"
        "4. Your ultimate goal is to produce a refined, synthesized legal report.\n"
        f"Please output in exactly the same format as:\n'''{syn_report_format}'''"
    )

    return synthesizer, prompt


def get_consensus_prompt_legal(domain, syn_report):
    """
    Trả về:
      - voter: Vai trò của mô hình (system role) khi đối chiếu tính nhất quán với báo cáo tổng hợp.
      - cons_prompt: Câu lệnh yêu cầu chuyên gia pháp lý cho biết có đồng thuận hay không.
    """
    voter = f"You are a legal expert specialized in the {domain} domain."
    
    cons_prompt = (
        f"Here is a legal report:\n{syn_report}\n\n"
        f"As a legal expert specialized in {domain}, please carefully read the report "
        f"and decide whether your own legal opinion is consistent with the contents of this report.\n"
        f"Please respond only with: [YES or NO]."
    )

    return voter, cons_prompt


def get_consensus_opinion_prompt_legal(domain, syn_report):
    """
    Trả về:
      - opinion_prompt: Câu lệnh yêu cầu chuyên gia pháp lý đề xuất chỉnh sửa cho báo cáo tổng hợp.
    """
    opinion_prompt = (
        f"Here is a legal report:\n{syn_report}\n\n"
        f"As a legal expert specialized in {domain}, please leverage your full expertise "
        f"to propose revisions or additional considerations to improve this report.\n"
        f"You should output in exactly the same format as:\n"
        f"'''Revisions: [proposed revision advice] '''"
    )

    return opinion_prompt


def get_revision_prompt_legal(syn_report, revision_advice):
    """
    Trả về:
      - revision_prompt: Câu lệnh yêu cầu LLM soạn lại báo cáo dựa trên các góp ý từ nhiều chuyên gia pháp lý.
    """
    revision_prompt = f"Here is the original legal report:\n{syn_report}\n\n"
    for domain, advice in revision_advice.items():
        revision_prompt += (
            f"Here is advice from a legal expert specialized in {domain}: {advice}\n"
        )
    revision_prompt += (
        "\nBased on the above advice, output the revised analysis in exactly the same format as:\n"
        "'''Total Analysis: [revised comprehensive legal analysis] '''"
    )

    return revision_prompt