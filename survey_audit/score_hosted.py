"""Build hosted requests or parse saved hosted responses for eligible Pew inputs."""

import argparse
import json
import math
import os
from pathlib import Path
import urllib.request

from survey_audit.prepare_data import read_rows

LETTER_INSTRUCTION = ('Return exactly one uppercase option letter and no other text. '
                      'Do not include spaces, punctuation, or an explanation.')
VERBAL_INSTRUCTION = ("Return a JSON object whose keys are exactly the uppercase option letters "
    "listed above and whose values are their probabilities for this person's answer. "
    'Use numbers between 0 and 1 that sum to 1, with at most four decimal places. '
    'Return every option and no other text.')


def letter_prompt(job, verbalized=False):
    if job['dataset'] != 'pew':
        raise ValueError('Hosted scoring accepts the public Pew source only')
    labels = {code: chr(65 + index) for index, code in enumerate(job['options'])}
    prefix = job['messages'][0]['content'].rsplit('### Response Options\n\n', 1)[0]
    options = '; '.join(f"{labels[code]} {job['option_labels'][code]}" for code in job['options'])
    instruction = VERBAL_INSTRUCTION if verbalized else LETTER_INSTRUCTION
    return prefix + '### Response Options\n\n' + options + '\n\n### Response\n\n' + instruction, labels


def request_body(job, provider, model):
    verbalized = provider == 'anthropic'
    prompt, labels = letter_prompt(job, verbalized)
    messages = [{'role': 'user', 'content': prompt}]
    if provider == 'openai':
        special = model == 'gpt-6.1-terra'
        body = dict(model=model, messages=messages, store=False, temperature=0,
                    max_completion_tokens=16 if special else 1, logprobs=True,
                    top_logprobs=5 if special else 20)
        if special:
            body.update(reasoning_effort='none', prompt_cache_options={'mode': 'explicit'})
    else:
        schema = {'type': 'object', 'properties': {label: {'type': 'number'} for label in labels.values()},
                  'required': list(labels.values()), 'additionalProperties': False}
        body = dict(model=model, messages=messages, max_tokens=128, thinking={'type': 'disabled'},
                    output_config={'effort': 'low', 'format': {'type': 'json_schema', 'schema': schema}})
    return body, labels


def normalize(values):
    if not values or any(isinstance(value, bool) or not isinstance(value, (int, float))
                         or not math.isfinite(value) or value < 0 or value > 1 for value in values.values()):
        raise ValueError('Invalid probability values')
    total = sum(values.values())
    if total <= 0:
        raise ValueError('Zero total probability')
    return {code: value / total for code, value in values.items()}


def parse_response(provider, labels, response):
    if provider == 'openai':
        choice = response['choices'][0]
        if choice['message'].get('refusal') or choice.get('finish_reason') not in ('stop', 'length'):
            raise ValueError('Refused, filtered, or invalid completion')
        entries = choice['logprobs']['content'][0]['top_logprobs']
        logs = {entry['token']: entry['logprob'] for entry in entries if entry['token'] in labels.values()}
        if not logs or any(not math.isfinite(value) for value in logs.values()):
            raise ValueError('No valid answer-label probabilities')
        largest = max(logs.values())
        values = {code: math.exp(logs[label] - largest) if label in logs else 0.0 for code, label in labels.items()}
        return normalize(values), dict(missing_labels=len(labels) - len(logs),
                                      raw_option_mass=sum(math.exp(value) for value in logs.values()))
    if response.get('stop_reason') != 'end_turn':
        raise ValueError('Incomplete verbalized probability response')
    content = ''.join(block['text'] for block in response['content'] if block['type'] == 'text')
    values = json.loads(content)
    if set(values) != set(labels.values()):
        raise ValueError('Verbalized probability keys differ from answer labels')
    values = normalize(values)
    return {code: values[label] for code, label in labels.items()}, {}


def call_provider(provider, body):
    if provider == 'openai':
        url = 'https://api.openai.com/v1/chat/completions'
        headers = {'Authorization': 'Bearer ' + os.environ['OPENAI_API_KEY']}
    else:
        url = 'https://api.anthropic.com/v1/messages'
        headers = {'x-api-key': os.environ['ANTHROPIC_API_KEY'], 'anthropic-version': '2026-10-21'}
    headers['Content-Type'] = 'application/json'
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method='POST')
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--provider', choices=('openai', 'anthropic'), required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--responses', type=Path, help='Saved response envelopes, keyed by cell_id')
    parser.add_argument('--execute', action='store_true', help='Make paid provider calls; otherwise export requests only')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.execute and args.responses:
        parser.error('Choose saved-response parsing or execution')
    saved = {row['cell_id']: row['response'] for row in read_rows(args.responses)} if args.responses else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open('w', encoding='utf-8') as stream:
        for job in read_rows(args.jobs):
            if not job.get('evaluation_cohort', True):
                continue
            if args.limit is not None and count >= args.limit:
                break
            body, labels = request_body(job, args.provider, args.model)
            cell_id = '/'.join((job['dataset'], str(job['respondent_id']), job['target'], job['condition']))
            if saved is None and not args.execute:
                record = dict(cell_id=cell_id, provider=args.provider, request=body)
            else:
                response = saved[cell_id] if saved is not None else call_provider(args.provider, body)
                probabilities, details = parse_response(args.provider, labels, response)
                record = {key: value for key, value in job.items() if key not in ('messages', 'scoring_texts')}
                record.update(model=args.model, probabilities=probabilities, scoring_details=details)
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            count += 1
    print(f'Processed {count} cells')


if __name__ == '__main__':
    main()
