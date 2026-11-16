"""Run the real Compose image against a local synthetic API, never a paid provider."""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.test_final_portability import FIXTURE_DOCUMENT, QUESTION, LocalModelServer


def main():
    if any((ROOT / name).exists() for name in ('.env', 'knowledge-base.zip')):
        raise RuntimeError('Use a clean staging checkout: smoke will not overwrite existing inputs')
    project = 'ai-lawyer-smoke-' + str(os.getpid())
    override = ROOT / '.local' / 'container-smoke-override.yaml'
    override.parent.mkdir(exist_ok=True)
    override.write_text('services:\n  chat:\n    extra_hosts:\n      - "host.docker.internal:host-gateway"\n', encoding='utf-8')
    command = ['docker', 'compose', '-p', project, '-f', str(ROOT / 'compose.yaml'), '-f', str(override)]

    def run(*args, timeout=600):
        result = subprocess.run([*command, *args], cwd=ROOT, capture_output=True, text=True,
                                encoding='utf-8', timeout=timeout)
        if result.returncode:
            raise RuntimeError(f'Compose command failed: {args!r}\n{result.stdout}\n{result.stderr}')
        return result.stdout

    def request(path, value=None):
        data = None if value is None else json.dumps(value, ensure_ascii=False).encode('utf-8')
        req = urllib.request.Request('http://127.0.0.1:8000' + path, data=data,
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=45) as response:
            content = response.read()
        return content.decode('utf-8') if path == '/' else json.loads(content)

    try:
        with LocalModelServer(host='0.0.0.0') as models, tempfile.TemporaryDirectory() as tmp:
            (ROOT / '.env').write_text('\n'.join([
                f'LLM_BASE_URL=http://host.docker.internal:{models.server.server_address[1]}',
                'LLM_API_KEY=synthetic-local-only', 'DEV_CACHE=0', 'MAX_RETRIES=0',
                'HTTP_TIMEOUT_S=5', 'QUESTION_BUDGET_S=30',
            ]) + '\n', encoding='utf-8')
            with zipfile.ZipFile(ROOT / 'knowledge-base.zip', 'w') as archive:
                archive.writestr('fictional-contract-2030.txt', FIXTURE_DOCUMENT)
            run('config', '--quiet')
            print(run('build', '--pull'), flush=True)
            print(run('up', '-d', '--wait', '--wait-timeout', '180'), flush=True)
            assert 'Юридический' in request('/')
            docs = request('/api/documents')
            assert len(docs) == 1, docs
            first = request('/api/chat', {'turns': [{'role': 'user', 'text': QUESTION}]})
            assert not first['fallback'] and '17 рабочих дней' in first['answer'], first
            assert first['sources'], first
            source = first['sources'][0]
            viewed = request('/api/documents/' + urllib.parse.quote(source['document']) +
                             '?node=' + urllib.parse.quote(source['node']))
            assert viewed['text'][viewed['start']:viewed['end']] == viewed['fragment']
            follow = request('/api/chat', {'turns': [
                {'role': 'user', 'text': QUESTION}, {'role': 'assistant', 'text': first['answer']},
                {'role': 'user', 'text': 'Повторите срок.'}]})
            assert not follow['fallback'] and '17 рабочих дней' in follow['answer'], follow
            uid = run('exec', '-T', 'chat', 'python', '-c', 'import os; print(os.getuid())').strip()
            assert uid == '10001', uid
            data = Path(tmp) / 'input'
            data.mkdir()
            (data / 'knowledge-base.zip').write_bytes((ROOT / 'knowledge-base.zip').read_bytes())
            with (data / 'questions.csv').open('w', encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['id', 'question'])
                writer.writerows([['fiction-1', QUESTION], ['fiction-2', QUESTION]])
            container = run('ps', '-q', 'chat').strip()
            subprocess.run(['docker', 'cp', str(data), container + ':/tmp/smoke-input'], check=True)
            run('exec', '-T', '-e', 'DATA_DIR=/tmp/smoke-input', 'chat', 'python', 'solution.py', timeout=90)
            output = run('exec', '-T', 'chat', 'cat', '/app/submission.csv')
            reader = csv.DictReader(io.StringIO(output))
            assert reader.fieldnames == ['id', 'answer']
            rows = list(reader)
            assert [row['id'] for row in rows] == ['fiction-1', 'fiction-2']
            assert all('17 рабочих дней' in row['answer'] and len(row['answer']) <= 10000 for row in rows)
            assert models.counts['/embeddings'] and models.counts['/rerank']
            summary = {'real_docker_compose': True, 'healthy': True, 'uid': int(uid),
                       'web_answer': True, 'source_highlight': True, 'history': True,
                       'batch_answers': len(rows), 'synthetic_kb': True, 'paid_calls': 0,
                       'model_requests': dict(models.counts)}
            print(json.dumps(summary), flush=True)
            (ROOT / '.local' / 'container-smoke-result.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    finally:
        try:
            print(run('logs', '--no-color', timeout=30), flush=True)
        finally:
            subprocess.run([*command, 'down', '--volumes', '--remove-orphans'], cwd=ROOT, timeout=60)
            for name in ('.env', 'knowledge-base.zip'):
                (ROOT / name).unlink(missing_ok=True)
            override.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
