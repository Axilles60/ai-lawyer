"""Final-delivery E2E: fresh fictional KB, no precomputed assets, local mock API only."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DOCUMENT = """Договор оказания услуг № ФТ-43/2030 от 01.03.2030
1. Предмет договора
1.1. Исполнитель передаёт отчёт заказчику.
2. Срок исполнения
2.1. Исполнитель передаёт отчёт в течение 17 рабочих дней после получения заявки.
3. Порядок передачи
3.1. Передача отчёта подтверждается подписанным актом.
"""
QUESTION = "По договору № ФТ-43/2030 в какой срок исполнитель передаёт отчёт?"


class LocalModelServer:
    def __init__(self, fail=False, host='127.0.0.1'):
        self.counts = Counter()
        self.lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                with owner.lock:
                    owner.counts[self.path] += 1
                status = 503 if fail else 200
                if fail:
                    result = {'error': 'Synthetic unavailable model'}
                elif self.path.endswith('/embeddings'):
                    result = {'data': []}
                    for i, value in enumerate(data['input']):
                        raw = hashlib.sha256(value.encode('utf-8')).digest()[:16]
                        vec = [float(v) - 127.5 for v in raw]
                        norm = math.sqrt(sum(v * v for v in vec))
                        result['data'].append({'index': i, 'embedding': [v / norm for v in vec]})
                elif self.path.endswith('/rerank'):
                    docs = data['documents']
                    result = {'results': [{'index': i, 'relevance_score': 1.0 - i / (len(docs) + 1)}
                                          for i in range(len(docs))]}
                else:
                    system = data['messages'][0]['content']
                    user = data['messages'][-1]['content']
                    if system.startswith('Ты — модуль разбора запросов'):
                        answer = {'scope': 'offtopic' if 'стихотворение' in user else 'legal',
                                  'query': QUESTION, 'doc_refs': ['ФТ-43/2030'],
                                  'node_refs': [], 'entities': [], 'subqueries': [], 'stepback': ''}
                    elif system.startswith('Ты — юридический ассистент'):
                        rows = re.findall(r'\[(Д\d+\.\d+)\]\s+([^\n]+)', user)
                        tag, quote = next((tag, text) for tag, text in rows if '17 рабочих дней' in text)
                        quote = quote.split(' ◀')[0].strip()
                        answer = {'decision': 'answer', 'claims': [
                            {'text': 'Отчёт передаётся в течение 17 рабочих дней после получения заявки [[' + tag + ']].',
                             'sources': [tag], 'quote': quote}], 'missing': ''}
                    elif 'best' in system:
                        answer = {'best': 1}
                    elif 'verdict' in system:
                        answer = {'verdict': 'ok', 'issues': []}
                    else:
                        answer = None
                    content = json.dumps(answer, ensure_ascii=False) if answer is not None else 'Срок передачи отчёта определяется договором.'
                    result = {'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}],
                              'usage': {'prompt_tokens': 1, 'completion_tokens': 1}}
                encoded = json.dumps(result, ensure_ascii=False).encode('utf-8')
                self.send_response(status)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        self.server = ThreadingHTTPServer((host, 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    @property
    def url(self):
        return f'http://127.0.0.1:{self.server.server_address[1]}'


def fixture(tmp: Path):
    source = Path(os.environ.get('FINAL_PORTABILITY_SOURCE', str(ROOT))).resolve()
    work, data = tmp / 'work', tmp / 'data'
    work.mkdir(); data.mkdir()
    for path in source.glob('*.py'):
        shutil.copy2(path, work / path.name)
    if (source / 'static').is_dir():
        shutil.copytree(source / 'static', work / 'static')
    with zipfile.ZipFile(data / 'knowledge-base.zip', 'w') as z:
        z.writestr('fictional-contract-2030.txt', FIXTURE_DOCUMENT)
    with (data / 'questions.csv').open('w', encoding='utf-8', newline='') as stream:
        out = csv.writer(stream)
        out.writerow(['id', 'question'])
        out.writerow(['fiction-1', QUESTION])
        out.writerow(['fiction-2', QUESTION])
    return work, data


def environment(data: Path, server: LocalModelServer):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('LLM_', 'OPENAI_', 'EMB', 'RERANK', 'USE_', 'DEV_'))
           and not k.upper().endswith('_PROXY')}
    env.update(DATA_DIR=str(data), KB_PATH=str(data / 'knowledge-base.zip'),
               LLM_BASE_URL=server.url, LLM_API_KEY='fixture-no-network',
               OPENAI_API_KEY='', EMBEDDER_API_KEY='fixture-no-network',
               RERANKER_API_KEY='fixture-no-network', DEV_CACHE='0',
               MAX_RETRIES='0', MAX_WORKERS='2', HTTP_TIMEOUT_S='5',
               QUESTION_BUDGET_S='30', NO_PROXY='127.0.0.1,localhost', PYTHONIOENCODING='utf-8')
    return env


class FinalPortabilityTests(unittest.TestCase):
    def test_source_view_uses_second_occurrence_actual_offsets(self):
        from fastapi.testclient import TestClient
        from unittest.mock import patch
        from kb_types import DocCard, Node, ParsedDoc
        import app
        text = 'Одинаковое положение.\nИной контекст.\nОдинаковое положение.'
        start = text.rfind('Одинаковое положение.')
        node = Node('second', 'd', None, 1, 'абз.3', 'абз.3',
                    text='Одинаковое положение.', start=start, end=len(text))
        card = DocCard('d', 'fiction.txt', 'fixture', title='Вымышленный документ')
        class Index:
            def __init__(self):
                self.parsed = [ParsedDoc(card, [node], text=text)]
            def doc(self, key): return card
            def node(self, key): return node
        with patch.object(app.app.state, 'index', Index(), create=True):
            viewed = TestClient(app.app).get('/api/documents/d?node=second').json()
        self.assertEqual(viewed['start'], start)
        self.assertEqual(text[viewed['start']:viewed['end']], node.text)
        self.assertEqual(node.start, start)

    def test_source_view_does_not_guess_ambiguous_normalized_location(self):
        from fastapi.testclient import TestClient
        from unittest.mock import patch
        from kb_types import DocCard, Node, ParsedDoc
        import app
        text = 'Повторённый фрагмент.\nПовторённый фрагмент.'
        node = Node('n', 'd', None, 1, 'абз.2', '', text='Повторённый фрагмент.')
        card = DocCard('d', 'fiction.txt', 'fixture')
        class Index:
            def __init__(self):
                self.parsed = [ParsedDoc(card, [node], text=text)]
            def doc(self, key): return card
            def node(self, key): return node
        with patch.object(app.app.state, 'index', Index(), create=True):
            viewed = TestClient(app.app).get('/api/documents/d?node=n').json()
        self.assertEqual((viewed['start'], viewed['end']), (-1, -1))
        self.assertEqual(viewed['fragment'], node.text)

    def test_solution_builds_index_and_answers_with_no_precomputed_assets(self):
        with tempfile.TemporaryDirectory() as folder, LocalModelServer() as server:
            work, data = fixture(Path(folder))
            self.assertFalse(any(work.glob('*.npz')))
            self.assertFalse(any(work.glob('*.sqlite')))
            run = subprocess.run([sys.executable, 'solution.py'], cwd=work, env=environment(data, server),
                                 capture_output=True, text=True, encoding='utf-8', timeout=80)
            self.assertEqual(run.returncode, 0, run.stderr[-2000:])
            with (work / 'submission.csv').open(encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, ['id', 'answer'])
                rows = list(reader)
            self.assertEqual([r['id'] for r in rows], ['fiction-1', 'fiction-2'])
            for row in rows:
                self.assertIn('17 рабочих дней', row['answer'])
                self.assertIn('2.1', row['answer'])
                self.assertLessEqual(len(row['answer']), 10000)
            self.assertGreater(server.counts['/embeddings'], 0)
            self.assertGreater(server.counts['/rerank'], 0)
            self.assertIn("'full': 2", run.stderr)

    def test_unavailable_api_still_produces_complete_submission(self):
        with tempfile.TemporaryDirectory() as folder, LocalModelServer(fail=True) as server:
            work, data = fixture(Path(folder))
            run = subprocess.run([sys.executable, 'solution.py'], cwd=work, env=environment(data, server),
                                 capture_output=True, text=True, encoding='utf-8', timeout=80)
            self.assertEqual(run.returncode, 0, run.stderr[-2000:])
            with (work / 'submission.csv').open(encoding='utf-8-sig', newline='') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([r['id'] for r in rows], ['fiction-1', 'fiction-2'])
            self.assertTrue(all(r['answer'].strip() for r in rows))

    def test_real_web_lifespan_chat_history_and_source_view(self):
        with tempfile.TemporaryDirectory() as folder, LocalModelServer() as server:
            work, data = fixture(Path(folder))
            script = r'''
from fastapi.testclient import TestClient
import app, json
with TestClient(app.app) as web:
    assert web.get('/').status_code == 200
    docs = web.get('/api/documents').json()
    assert len(docs) == 1
    first = web.post('/api/chat', json={'turns':[{'role':'user','text':QUESTION}]}).json()
    assert not first['fallback'], first
    assert '17 рабочих дней' in first['answer'], first
    assert first['sources'], first
    source = first['sources'][0]
    viewed = web.get('/api/documents/' + source['document'], params={'node':source['node']}).json()
    assert '17 рабочих дней' in viewed['text']
    if viewed['fragment']:
        assert viewed['start'] >= 0
        assert viewed['text'][viewed['start']:viewed['end']] == viewed['fragment']
    follow = web.post('/api/chat', json={'turns':[{'role':'user','text':QUESTION},
        {'role':'assistant','text':first['answer']},{'role':'user','text':'Повторите срок.'}]}).json()
    assert not follow['fallback'], follow
    assert '17 рабочих дней' in follow['answer']
print(json.dumps({'web_lifespan':True,'chat':True,'history':True,'source_view':True}))
'''
            run = subprocess.run([sys.executable, '-c', 'QUESTION=' + repr(QUESTION) + '\n' + script],
                                 cwd=work, env=environment(data, server), capture_output=True,
                                 text=True, encoding='utf-8', timeout=80)
            self.assertEqual(run.returncode, 0, run.stderr[-3000:])
            result = json.loads(run.stdout.strip().splitlines()[-1])
            self.assertTrue(all(result.values()))
            self.assertGreater(server.counts['/embeddings'], 0)


class FrontEndBehaviorTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Node.js is optional for browser-script regression checks')
    def test_unicode_source_race_and_failed_chat_retry(self):
        # Execute the shipped page script with a local DOM/fetch harness: no browser
        # package or external endpoint is needed for these state regressions.
        script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
class Element {
  constructor(tag='div'){this.tagName=tag;this.children=[];this._text='';this.value='';this.disabled=false;this.scrollTop=0;this.scrollHeight=0;}
  get textContent(){return this._text+this.children.map(c=>c.textContent).join('');}
  set textContent(text){this._text=String(text);this.children=[];}
  append(...nodes){for(const node of nodes){node.parentElement=this;this.children.push(node);}}
  replaceChildren(...nodes){this._text='';this.children=[];this.append(...nodes);}
  scrollIntoView(){this.scrolled=true;}
  focus(){this.focused=true;}
}
const elements=Object.fromEntries(['messages','documents','documentText','status','question','send','clear','form','docTitle'].map(id=>[id,new Element()]));
const document={querySelector:s=>elements[s.slice(1)],createElement:t=>new Element(t),createTextNode:t=>{const n=new Element('#text');n.textContent=t;return n}};
const prefix='Начало 😀\n'+'Длинный документ.\n'.repeat(120),fragment='Точный подтверждающий фрагмент.';
const source={title:'Unicode source',text:prefix+fragment+'\nКонец',fragment,start:Array.from(prefix).length,end:Array.from(prefix+fragment).length};
const response=d=>({ok:true,json:async()=>d});
let resolveSlow,resolveChat,mode='hold',requests=[];
async function fetch(url,options){
 if(url==='/api/documents')return response([{id:'unicode',title:'Unicode source'}]);
 if(url.startsWith('/api/documents/unicode?'))return response(source);
 if(url.startsWith('/api/documents/slow?'))return new Promise(resolve=>resolveSlow=()=>resolve(response({title:'Old',text:'Old source',start:-1,end:-1,fragment:''})));
 if(url.startsWith('/api/documents/fast?'))return response({title:'Latest',text:'Latest source',start:-1,end:-1,fragment:''});
 if(url.startsWith('/api/documents/bad?'))throw Error('offline');
 if(url==='/api/chat'){
  requests.push(JSON.parse(options.body));
  if(mode==='fail')throw Error('offline');
  if(mode==='hold')await new Promise(resolve=>resolveChat=resolve);
  return response({answer:'Подтвержденный ответ',sources:[],fallback:false});
 }
 throw Error('Unexpected external fetch '+url);
}
const context=vm.createContext({document,fetch,console});
const html=fs.readFileSync('static/index.html','utf8');
vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1],context);
(async()=>{
 await context.openDoc('unicode','n');
 const mark=elements.documentText.children.find(n=>n.tagName==='mark');
 assert(mark,'source must be highlighted');assert.equal(mark.textContent,fragment);assert(mark.scrolled,'citation must be brought into view');
 const old=context.openDoc('slow');await context.openDoc('fast');resolveSlow();await old;
 assert.equal(elements.docTitle.textContent,'Latest');assert.equal(elements.documentText.textContent,'Latest source');
 await context.openDoc('bad');assert.match(elements.documentText.textContent,/Не удалось загрузить источник/);assert(!elements.documentText.textContent.includes('Latest source'));
 elements.question.value='Первый вопрос';const pending=elements.form.onsubmit({preventDefault(){}});
 assert.match(elements.status.textContent,/Ищу/);assert(elements.send.disabled);assert(elements.clear.disabled);assert(elements.question.disabled);
 await elements.form.onsubmit({preventDefault(){}});assert.equal(requests.length,1,'pending submission must not duplicate request');
 resolveChat();await pending;assert.equal(elements.status.textContent,'Готово');
 mode='fail';elements.question.value='Вопрос для повтора';await elements.form.onsubmit({preventDefault(){}});
 assert.equal(elements.question.value,'Вопрос для повтора');assert(!elements.send.disabled);assert(!elements.question.disabled);
 mode='ok';await elements.form.onsubmit({preventDefault(){}});
 assert.deepStrictEqual(requests[2].turns.map(t=>t.role),['user','assistant','user']);
 assert.equal(requests[2].turns[2].text,'Вопрос для повтора');
 elements.clear.onclick();assert.equal(elements.messages.children.length,0);assert.equal(elements.question.value,'');
 elements.question.value='Новый диалог';await elements.form.onsubmit({preventDefault(){}});assert.equal(requests[3].turns.length,1);
 console.log('frontend state, Unicode source, race, and retry: OK');
})().catch(e=>{console.error(e);process.exitCode=1});
"""
        result = subprocess.run([shutil.which('node'), '-'], input=script, cwd=ROOT,
                                capture_output=True, text=True, encoding='utf-8', timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('frontend state, Unicode source, race, and retry: OK', result.stdout)


if __name__ == '__main__':
    unittest.main()
