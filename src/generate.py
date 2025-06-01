from prompt import *
from llm import get_llm_output
from utils import *

def fully_decode(question, options, gold_answer, args):
    question_domains, options_domains = [], []
    question_analyses, option_analyses = {}, {}
    syn_report, output = "", ""
    vote_history, revision_history, syn_repo_history = [], [], []

    if args.method == "base_direct":
        direct_prompt = direct_prompting(question, options)
        output = get_llm_output(
            temperature=0,
            max_tokens=50,
            system_role="",
            prompt=direct_prompt
        )
        ans, output = cleansing_final_output(output)

    elif args.method == "base_cot":
        cot_prompt = cot_prompting(question, options)
        output = get_llm_output(
            temperature=0,
            max_tokens=300,
            system_role="",
            prompt=cot_prompt
        )
        ans, output = cleansing_final_output(output)

    else:
        # 1. Phân loại question domains (NUM_LD subfields)
        question_classifier, prompt_qd = get_question_domains_prompt_legal(question)
        raw_q_domains = get_llm_output(
            temperature=0,
            max_tokens=50,
            system_role=question_classifier,
            prompt=prompt_qd
        )
        if raw_q_domains.strip() == "ERROR.":
            raw_q_domains = "Legal Field: " + " | ".join(["General Law" for _ in range(NUM_LD)])
        question_domains = raw_q_domains.split(":")[-1].strip().split(" | ")

        # 2. Phân loại options domains (NUM_LOD subfields)
        options_classifier, prompt_od = get_options_domains_prompt_legal(question, options)
        raw_o_domains = get_llm_output(
            temperature=0,
            max_tokens=50,
            system_role=options_classifier,
            prompt=prompt_od
        )
        if raw_o_domains.strip() == "ERROR.":
            raw_o_domains = "Legal Field: " + " | ".join(["General Law" for _ in range(NUM_LOD)])
        options_domains = raw_o_domains.split(":")[-1].strip().split(" | ")

        # 3. Lấy question analyses từ từng domain
        tmp_q_analyses = []
        for dom in question_domains:
            q_analyzer, prompt_qa = get_question_analysis_prompt_legal(question, dom)
            raw_q_analysis = get_llm_output(
                temperature=0,
                max_tokens=300,
                system_role=q_analyzer,
                prompt=prompt_qa
            )
            tmp_q_analyses.append(raw_q_analysis)
        question_analyses = cleansing_analysis(tmp_q_analyses, question_domains, "question")

        # 4. Lấy option analyses từ từng option domain
        tmp_o_analyses = []
        for dom in options_domains:
            o_analyzer, prompt_oa = get_options_analysis_prompt_legal(
                question, options, dom, question_analyses
            )
            raw_o_analysis = get_llm_output(
                temperature=0,
                max_tokens=300,
                system_role=o_analyzer,
                prompt=prompt_oa
            )
            tmp_o_analyses.append(raw_o_analysis)
        option_analyses = cleansing_analysis(tmp_o_analyses, options_domains, "option")

        # 5. Nếu chỉ cần phân tích và trả lời trực tiếp
        if args.method == "anal_only":
            answer_prompt = get_final_answer_prompt_analonly_legal(
                question, options, question_analyses, option_analyses
            )
            output = get_llm_output(
                temperature=0,
                max_tokens=2500,
                system_role="",
                prompt=answer_prompt
            )
            ans, output = cleansing_final_output(output)

        else:
            # 6. Tổng hợp báo cáo (synthesized report)
            q_analyses_text = transform_dict2text(question_analyses, "question", question)
            o_analyses_text = transform_dict2text(option_analyses, "options", options)
            synthesizer, prompt_syn = get_synthesized_report_prompt_legal(
                q_analyses_text, o_analyses_text
            )
            raw_syn = get_llm_output(
                temperature=0,
                max_tokens=2500,
                system_role=synthesizer,
                prompt=prompt_syn
            )
            syn_report = cleansing_syn_report(question, options, raw_syn)

            # 7. Nếu chỉ cần tổng hợp rồi trả lời
            if args.method == "syn_only":
                answer_prompt = get_final_answer_prompt_wsyn_legal(syn_report)
                output = get_llm_output(
                    temperature=0,
                    max_tokens=2500,
                    system_role="",
                    prompt=answer_prompt
                )
                ans, output = cleansing_final_output(output)

            # 8. Nếu cần bỏ phiếu và chỉnh sửa lặp lại
            elif args.method == "syn_verif":
                all_domains = question_domains + options_domains
                syn_repo_history = [syn_report]

                hasno_flag = True
                num_try = 0

                while num_try < args.max_attempt_vote and hasno_flag:
                    domain_opinions = {}
                    revision_advice = {}
                    num_try += 1
                    hasno_flag = False

                    # Mỗi domain vote và nếu không đồng ý, thu advice
                    for dom in all_domains:
                        voter, cons_prompt = get_consensus_prompt_legal(dom, syn_report)
                        raw_opi = get_llm_output(
                            temperature=0,
                            max_tokens=30,
                            system_role=voter,
                            prompt=cons_prompt
                        )
                        domain_opinion = cleansing_voting(raw_opi)  # "yes" / "no"
                        domain_opinions[dom] = domain_opinion

                        if domain_opinion.lower() == "no":
                            advice_prompt = get_consensus_opinion_prompt_legal(dom, syn_report)
                            advice_out = get_llm_output(
                                temperature=0,
                                max_tokens=500,
                                system_role=voter,
                                prompt=advice_prompt
                            )
                            revision_advice[dom] = advice_out
                            hasno_flag = True

                    if hasno_flag:
                        revision_prompt = get_revision_prompt_legal(syn_report, revision_advice)
                        revised = get_llm_output(
                            temperature=0,
                            max_tokens=2500,
                            system_role="",
                            prompt=revision_prompt
                        )
                        syn_report = cleansing_syn_report(question, options, revised)
                        revision_history.append(revision_advice)
                        syn_repo_history.append(syn_report)

                    vote_history.append(domain_opinions)

                # Cuối cùng: trả lời từ bản báo cáo tổng hợp
                answer_prompt = get_final_answer_prompt_wsyn_legal(syn_report)
                output = get_llm_output(
                    temperature=0,
                    max_tokens=2500,
                    system_role="",
                    prompt=answer_prompt
                )
                ans, output = cleansing_final_output(output)

    data_info = {
        "question": question,
        "options": options,
        "pred_answer": ans,
        "gold_answer": gold_answer,
        "question_domains": question_domains,
        "option_domains": options_domains,
        "question_analyses": question_analyses,
        "option_analyses": option_analyses,
        "syn_report": syn_report,
        "vote_history": vote_history,
        "revision_history": revision_history,
        "syn_repo_history": syn_repo_history,
        "raw_output": output
    }

    return data_info

