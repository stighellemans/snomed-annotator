"""HTTP contract tests: provider routing, credential isolation and schema checks.

No cloud calls, model downloads or patient data are used.
"""
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch

from app import llm, pipeline

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}


class ProviderContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.requests = []
        cls.reply = {"ok": True}
        cls.http_status = 200
        cls.finish_reason = "stop"
        cls.response_status = "completed"

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                cls.requests.append((self.path, dict(self.headers), body))
                text = json.dumps(cls.reply)
                if cls.http_status != 200:
                    result = {"error": {"message": "unsupported format", "type": "invalid_request_error"}}
                elif self.path.endswith('/responses'):
                    result = {"id": "resp_test", "object": "response", "created_at": 1,
                              "status": cls.response_status, "model": "test-model",
                              "output": [{"type": "message", "id": "msg_test", "role": "assistant", "status": "completed",
                                          "content": [{"type": "output_text", "text": text, "annotations": []}]}]}
                elif self.path.endswith('/messages'):
                    name = body['tool_choice']['name']
                    result = {"id": "msg_test", "stop_reason": "tool_use", "content": [{"type": "tool_use", "name": name, "input": cls.reply}]}
                else:
                    result = {"id": "chat_test", "object": "chat.completion", "created": 1,
                              "model": "test-model", "choices": [{"index": 0, "finish_reason": cls.finish_reason,
                              "message": {"role": "assistant", "content": text}}]}
                encoded = json.dumps(result).encode()
                self.send_response(cls.http_status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.endpoint = f'http://127.0.0.1:{cls.server.server_port}/v1'

    @classmethod
    def tearDownClass(cls):
        llm._llm_client.cache_clear()
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.__class__.requests.clear()
        self.__class__.reply = {"ok": True}
        self.__class__.http_status = 200
        self.__class__.finish_reason = "stop"
        self.__class__.response_status = "completed"
        config = replace(llm.settings, llm_provider='openai_compatible', llm_base_url=self.endpoint,
                         llm_api_key='provider-key', openai_api_key='separate-embedding-key',
                         llm_model='test-model', llm_api='', llm_json_mode='json_schema', llm_reasoning_effort='')
        self.config = config
        self.patcher = patch.object(llm, 'settings', config)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def call(self):
        return llm.call_structured('Return the requested JSON.', 'Synthetic test.', 'result', SCHEMA)

    def test_chat_endpoint_and_provider_key(self):
        self.assertEqual(self.call(), {'ok': True})
        path, headers, body = self.requests[-1]
        self.assertEqual(path, '/v1/chat/completions')
        self.assertEqual(headers['Authorization'], 'Bearer provider-key')
        self.assertNotIn('reasoning_effort', body)
        self.assertEqual(body['response_format']['json_schema']['schema'], SCHEMA)

    def test_compatible_endpoint_never_inherits_openai_key(self):
        with patch.object(llm, 'settings', replace(self.config, llm_api_key='')):
            self.call()
        self.assertEqual(self.requests[-1][1]['Authorization'], 'Bearer local')

    def test_responses_api_keeps_its_own_schema_shape(self):
        with patch.object(llm, 'settings', replace(self.config, llm_provider='openai', llm_api='responses', llm_reasoning_effort='low')):
            self.assertEqual(self.call(), {'ok': True})
        path, _, body = self.requests[-1]
        self.assertEqual(path, '/v1/responses')
        self.assertEqual(body['text']['format']['schema'], SCHEMA)
        self.assertEqual(body['reasoning'], {'effort': 'low'})

    def test_native_anthropic_uses_tool_schema_and_separate_auth(self):
        with patch.object(llm, 'settings', replace(self.config, llm_provider='anthropic', llm_reasoning_effort='low')):
            self.assertEqual(self.call(), {'ok': True})
        path, headers, body = self.requests[-1]
        self.assertEqual(path, '/v1/messages')
        self.assertEqual(headers['x-api-key'], 'provider-key')
        self.assertNotIn('Authorization', headers)
        self.assertEqual(body['tools'][0]['input_schema'], SCHEMA)
        self.assertNotIn('reasoning', body)
        self.assertNotIn('reasoning_effort', body)

    def test_json_mode_and_prompt_mode_still_validate_output(self):
        for mode in ['json_object', 'prompt']:
            with self.subTest(mode=mode), patch.object(llm, 'settings', replace(self.config, llm_json_mode=mode)):
                self.assertEqual(self.call(), {'ok': True})
                body = self.requests[-1][2]
                self.assertIn('schema', body['messages'][0]['content'])
                self.assertEqual('response_format' in body, mode == 'json_object')
                self.__class__.reply = {'ok': 'not-a-boolean'}
                with self.assertRaisesRegex(RuntimeError, 'schema'):
                    self.call()
                self.__class__.reply = {'ok': True}

    def test_api_rejection_does_not_trigger_provider_or_mode_fallback(self):
        self.__class__.http_status = 400
        with self.assertRaisesRegex(RuntimeError, 'HTTP 400'):
            self.call()
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.requests[0][0], '/v1/chat/completions')

    def test_truncated_and_incomplete_outputs_are_rejected(self):
        self.__class__.finish_reason = 'length'
        with self.assertRaisesRegex(RuntimeError, 'afgebroken'):
            self.call()
        with patch.object(llm, 'settings', replace(self.config, llm_api='responses')):
            self.__class__.response_status = 'incomplete'
            with self.assertRaisesRegex(RuntimeError, 'onvolledig'):
                self.call()

    def test_invalid_json_and_additional_fields_are_rejected(self):
        for text in ['not JSON', '{"ok": NaN}', '{"ok": true, "extra": 1}']:
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                llm._decode(text, SCHEMA, 'result')
        with self.assertRaises(RuntimeError):
            llm._decode('{"value": 1e999}', {'type': 'object', 'properties': {'value': {'type': 'number'}}}, 'result')

    def test_embedding_client_uses_only_the_openai_endpoint_and_key(self):
        llm.get_embedding_client.cache_clear()
        client = llm.get_embedding_client()
        self.assertEqual(client.api_key, 'separate-embedding-key')
        self.assertEqual(str(client.base_url), 'https://api.openai.com/v1/')
        llm.get_embedding_client.cache_clear()

    def test_cache_separates_same_model_at_different_providers_and_endpoints(self):
        first = pipeline._result_key('Synthetic note', True, 'biolord')
        for changed in [replace(self.config, llm_provider='openai'),
                        replace(self.config, llm_base_url=self.endpoint + '/other'),
                        replace(self.config, llm_json_mode='json_object'),
                        replace(self.config, llm_reasoning_effort='high')]:
            with self.subTest(changed=changed.llm_provider), patch.object(llm, 'settings', changed):
                self.assertNotEqual(first, pipeline._result_key('Synthetic note', True, 'biolord'))

    def test_extraction_cache_does_not_reuse_output_from_another_provider(self):
        class Cache:
            def __init__(self): self.data = {}
            def get(self, key): return self.data.get(key)
            def set(self, key, value): self.data[key] = value
        with patch.object(pipeline, '_ner_cache', Cache()), patch.object(pipeline, 'call_structured', return_value={'spans': []}) as call:
            pipeline.extract_spans('Synthetic note', 40)
            pipeline.extract_spans('Synthetic note', 40)
            with patch.object(llm, 'settings', replace(self.config, llm_provider='anthropic')):
                pipeline.extract_spans('Synthetic note', 40)
            self.assertEqual(call.call_count, 2)

    def test_readiness_reports_missing_setup_without_network_calls(self):
        from fastapi.testclient import TestClient
        from app import main
        with patch.object(main, 'configuration_errors', return_value=['Missing model']), patch.object(main, 'embedder_available', return_value=False):
            response = TestClient(main.app).get('/api/readiness')
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()['detail'][0], 'Missing model')
        self.assertEqual(self.requests, [])

    def test_concept_outside_retrieved_candidates_is_not_linked(self):
        class Store:
            def search(self, *args, **kwargs):
                return [{'concept_id': '38341003', 'fsn': 'Hypertension', 'semantic_tag': 'disorder', 'vector_score': 0.9}]
            def concept_label(self, concept_id):
                return 'Hypertension'
        class EmptyCache:
            def get(self, key): return None
            def set(self, key, value): pass
        span = {'text': 'hypertensie', 'search_term': 'hypertension', 'category': 'disorder'}
        wrong = {'status': 'linked', 'concept_id': 'invented', 'term': 'Invented', 'confidence': 1.0}
        with patch.object(pipeline, '_result_cache', EmptyCache()), patch.object(pipeline, 'get_store', return_value=Store()), patch.object(pipeline, 'extract_spans', return_value=[span]), patch.object(pipeline, 'choose_best_concept', return_value=wrong):
            result = pipeline.annotate('hypertensie', True, 'biolord')
        self.assertEqual(result['spans'][0]['status'], 'none')
        self.assertEqual(result['spans'][0]['concept_id'], '')


if __name__ == '__main__':
    unittest.main()
