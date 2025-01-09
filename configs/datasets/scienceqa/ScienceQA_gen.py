from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets.scienceqa import ScienceQADataset
from opencompass.utils.text_postprocessors import first_capital_postprocess


scienceqa_reader_cfg = dict(
    input_columns=['lecture', 'input', 'A', 'B', 'C', 'D'],
    output_column='target',
    train_split='train')


scienceqa_datasets = []

scienceqa_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=
                f'{{lecture}}\nQuestion: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nAnswer: '
            ),
            dict(role='BOT', prompt='{target}\n')
        ]),
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin='</E>',
            round=[
                dict(
                    role='HUMAN',
                    prompt=
                    f'{{lecture}}\nQuestion: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nAnswer: '
                ),
            ],
        ),
        ice_token='</E>',
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
    inferencer=dict(type=GenInferencer),
)

scienceqa_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(type=first_capital_postprocess))

scienceqa_datasets.append(
    dict(
        abbr=f'ScienceQA',
        type=ScienceQADataset,
        path='./data/ScienceQA_text_only',
        name='scienceqa',
        reader_cfg=scienceqa_reader_cfg,
        infer_cfg=scienceqa_infer_cfg,
        eval_cfg=scienceqa_eval_cfg,
    ))
