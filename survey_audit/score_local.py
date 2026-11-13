"""Score every token in complete option-code JSON answers with a local checkpoint."""

import argparse
import importlib.metadata
import json
import math
from pathlib import Path

from survey_audit.prepare_data import read_rows


def answer_texts(job, answer_format):
    """Preserve saved candidates or apply the run's recorded JSON serializer."""
    if answer_format == 'saved':
        return job['scoring_texts']
    separators = (',', ':') if answer_format == 'compact' else (', ', ': ')
    return {code: json.dumps({'option': str(code)}, ensure_ascii=False,
                             separators=separators) for code in job['options']}


def score_options(model, tokenizer, job, torch, template_arguments=None):
    """Separate prompt/candidate tokenization; sum candidate log probabilities."""
    rendered = tokenizer.apply_chat_template(job['messages'], tokenize=False, add_generation_prompt=True, **(template_arguments or {}))
    prompt = tokenizer.encode(rendered, add_special_tokens=False)
    candidates = [tokenizer.encode(job['scoring_texts'][code], add_special_tokens=False) for code in job['options']]
    maximum = max(len(prompt) + len(candidate) for candidate in candidates)
    padding = tokenizer.pad_token_id
    if padding is None:
        padding = tokenizer.eos_token_id
    inputs = torch.full((len(candidates), maximum), padding, dtype=torch.long, device=model.device)
    attention = torch.zeros_like(inputs)
    batch_positions, token_positions, targets = [], [], []
    for index, candidate in enumerate(candidates):
        sequence = prompt + candidate
        start = maximum - len(sequence)
        inputs[index, start:] = torch.tensor(sequence, dtype=torch.long, device=model.device)
        attention[index, start:] = 1
        for offset, token in enumerate(candidate):
            batch_positions.append(index)
            token_positions.append(start + len(prompt) + offset - 1)
            targets.append(token)
    positions = attention.long().cumsum(-1) - 1
    positions.masked_fill_(attention == 0, 0)
    with torch.inference_mode():
        outputs = model(input_ids=inputs, attention_mask=attention, position_ids=positions, use_cache=False)
        logits = outputs.logits[batch_positions, token_positions].float()
        selected = logits.gather(1, torch.tensor(targets, dtype=torch.long, device=model.device).unsqueeze(1)).squeeze(1)
        token_logs = (selected - torch.logsumexp(logits, dim=1)).detach().cpu().tolist()
    scores, offset = [], 0
    for candidate in candidates:
        scores.append(float(sum(token_logs[offset:offset + len(candidate)])))
        offset += len(candidate)
    largest = max(scores)
    mass = [math.exp(value - largest) for value in scores]
    total = sum(mass)
    return ({code: value / total for code, value in zip(job['options'], mass)},
            dict(zip(job['options'], scores)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Complete local instruction-model snapshot')
    parser.add_argument('--revision', required=True, help='Exact checkpoint revision for the runtime record')
    parser.add_argument('--model', required=True, help='Public model identifier recorded in predictions')
    parser.add_argument('--dtype', choices=('float32', 'float16', 'bfloat16'), default='bfloat16')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--device-map', choices=('auto',), help='Distribute a large checkpoint over available devices')
    parser.add_argument('--chat-template-from', type=Path, help='Local tokenizer snapshot supplying the approved template')
    parser.add_argument('--disable-thinking', action='store_true', help='Pass enable_thinking=False to the native template')
    parser.add_argument('--attention', choices=('eager', 'sdpa'), default='sdpa')
    parser.add_argument('--answer-format', choices=('saved', 'spaced', 'compact'), default='saved',
                        help='Complete JSON candidate serialization; saved uses the job strings')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    configuration = json.loads((args.checkpoint / 'config.json').read_text())
    if configuration.get('quantization_config'):
        raise ValueError('Use the complete unquantized instruction checkpoint')
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=False)
    if args.chat_template_from:
        template_tokenizer = AutoTokenizer.from_pretrained(args.chat_template_from, local_files_only=True, trust_remote_code=False)
        tokenizer.chat_template = template_tokenizer.chat_template
    template_arguments = {}
    if args.disable_thinking or 'Qwen3.5-' in args.model or 'gemma-4-' in args.model:
        template_arguments['enable_thinking'] = False
    loading = dict(local_files_only=True, trust_remote_code=False,
                   torch_dtype=getattr(torch, args.dtype), low_cpu_mem_usage=True, attn_implementation=args.attention)
    if args.device_map:
        loading['device_map'] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, **loading)
    if not args.device_map:
        model.to(args.device)
    model.eval()
    runtime = dict(model=args.model, revision=args.revision, dtype=args.dtype,
        quantization='none', attention=args.attention, device=args.device, device_map=args.device_map,
        template_arguments=template_arguments, template_source='explicit_override' if args.chat_template_from else 'checkpoint',
        temperature=0, decoding='none; complete answer sequence scoring', answer_format=args.answer_format,
        prompt_special_tokens=False, candidate_special_tokens=False, eos_scored=False,
        score='sum of candidate token log probabilities; softmax over allowed answers',
        versions={name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'accelerate')})
    args.runtime.parent.mkdir(parents=True, exist_ok=True)
    args.runtime.write_text(json.dumps(runtime, indent=2) + '\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open('w', encoding='utf-8') as stream:
        for job in read_rows(args.jobs):
            if not job.get('evaluation_cohort', True):
                continue
            if args.limit is not None and count >= args.limit:
                break
            job['scoring_texts'] = answer_texts(job, args.answer_format)
            probabilities, scores = score_options(model, tokenizer, job, torch, template_arguments)
            record = {key: value for key, value in job.items() if key not in ('messages', 'scoring_texts')}
            record.update(model=args.model, probabilities=probabilities, sequence_log_probabilities=scores)
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            count += 1
    print(f'Scored {count} cells')


if __name__ == '__main__':
    main()
