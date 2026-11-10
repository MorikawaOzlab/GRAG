import unittest
from babyai.task import parse_goal, validate_goal, evaluate_goal
from babyai.ppo_agent import PPOAgent
from babyai.ollama_agent import ACTIONS


class Tests(unittest.TestCase):
    def test_decoded_view_orientation(self):
        from babyai.ollama_agent import describe_view
        expected = [(2,1), (-1,2), (-2,-1), (1,-2)]
        expected_right = ['south','west','north','east']
        image = [[[1,0,0] for _ in range(7)] for _ in range(7)]
        image[4][4] = [6,0,0]  # Red ball: two ahead, one right.
        image[3][6] = [5,2,0]  # Carried key is not an object on the floor.
        for direction in range(4):
            with self.subTest(direction=direction):
                view = describe_view({'image':image, 'direction':direction})
                self.assertEqual(view['rows_top_to_bottom'][4][4], 'red ball')
                self.assertEqual(len(view['visible_objects']), 1)
                obj = view['visible_objects'][0]
                self.assertEqual(obj['relative_offset'], {'forward':2,'right':1})
                self.assertEqual(obj['absolute_offset'], {'east':expected[direction][0], 'south':expected[direction][1]})
                self.assertEqual(view['orientation']['right'], expected_right[direction])

    def test_action_markdown_and_ambiguity(self):
        from babyai.ollama_agent import parse_action
        for content in (
            '赤いボールに向かうために、まず前方（西）へ進むことが最適です。 \n**action: forward**',
            '説明です。\n**ACTION:** **forward**',
            '説明です。\nACTION: `forward`',
            '```text\nACTION: forward\n```',
            'ACTION： forward',
            'forward',
        ):
            with self.subTest(content=content):
                self.assertEqual(parse_action(content), 'forward')
        for content in ('forward or left', 'ACTION: fly', 'ACTION: left\nACTION: right',
                        'ACTION: left\nACTION: invalid', 'I might move forward.'):
            with self.subTest(content=content), self.assertRaises(ValueError):
                parse_action(content)

    def test_color_optional_missions(self):
        goal = parse_goal('open the door')
        self.assertEqual(goal, {'type': 'open', 'object': {'type': 'door'}})
        state = {'objects': [{'type': 'door', 'color': 'blue', 'is_open': False}]}
        self.assertFalse(evaluate_goal(goal, state, {}))
        state['objects'].append({'type': 'door', 'color': 'red', 'is_open': True})
        self.assertTrue(evaluate_goal(goal, state, {}))
        self.assertFalse(evaluate_goal(parse_goal('open the blue door'), state, {}))
        self.assertEqual(parse_goal('  OPEN   THE DOOR '), goal)
        self.assertEqual(parse_goal('pick up a key')['object'], {'type': 'key'})
        self.assertEqual(parse_goal('go to the ball')['object'], {'type': 'ball'})

    def test_goal_rules(self):
        state={'objects':[{'type':'door','color':'blue','position':[1,1],'is_open':True},{'type':'key','color':'red','position':[1,2]}], 'front':[1,1], 'carrying':{'type':'key','color':'red'}}
        a=parse_goal('open the blue door')
        b=parse_goal('pick up the red key')
        self.assertTrue(evaluate_goal(a,state,{}))
        self.assertTrue(evaluate_goal({'type':'and','goals':[a,b]},state,{}))
        memory={}
        sequence={'type':'sequence','goals':[a,b]}
        self.assertFalse(evaluate_goal(sequence,state,memory))
        self.assertTrue(evaluate_goal(sequence,state,memory))
        self.assertTrue(evaluate_goal({'type':'putnext','object':{'type':'key'},'target':{'type':'door'}},state,{}))
        with self.assertRaises(ValueError):
            validate_goal({'type':'unknown'})

    def test_ppo_update_checkpoint(self):
        import tempfile
        import torch
        from pathlib import Path
        obs={'image':[[[1,0,0]]*7 for _ in range(7)],'direction':0,'mission':'open the blue door'}
        agent=PPOAgent(obs)
        before=[p.detach().clone() for p in agent.actor.parameters()]
        for i in range(8):
            action=agent.choose_action(obs)
            self.assertIn(action,ACTIONS)
            agent.learn(obs,action,float(i==7),obs,i==7,False)
        agent.update()
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,agent.actor.parameters())))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'agent.pt'
            agent.save(path)
            restored=PPOAgent(obs)
            restored.load(path)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(agent.actor.parameters(),restored.actor.parameters())))



class IntegrationTests(unittest.TestCase):
    def test_old_cumulative_llm_steps_are_read_as_stage_steps(self):
        from babyai.app import Session
        session = Session.__new__(Session)
        session.history = [{'total_steps':20}, {'total_steps':35}]
        for total, episode, step in ((20,1,20), (21,2,1), (35,2,15), (36,3,1)):
            normalized = session.normalize_llm_trace({'step':total})
            self.assertEqual(normalized, {'step':step, 'episode':episode, 'total_steps':total})
        trace = {'step':1, 'episode':2, 'total_steps':21}
        self.assertEqual(session.normalize_llm_trace(trace), trace)

    def test_wall_collision_feedback_and_llm_memory(self):
        import importlib.util
        if not importlib.util.find_spec('minigrid'):
            self.skipTest('MiniGrid is not installed')
        import io
        import json
        from unittest.mock import patch
        from babyai.environment import BabyAIEnvironment
        from babyai.ollama_agent import OllamaAgent, describe_front
        world = BabyAIEnvironment(max_steps=8)
        try:
            world.reset()
            env = world.env.unwrapped
            from babyai.ollama_agent import describe_view
            image = [[[1,0,0] for _ in range(7)] for _ in range(7)]
            image[4][4] = [6,0,0]
            for direction in range(4):
                env.agent_pos, env.agent_dir = (3,3), direction
                obj = describe_view({'image':image,'direction':direction})['visible_objects'][0]
                offset = obj['absolute_offset']
                self.assertEqual(env.get_view_coords(3+offset['east'],3+offset['south']), (4,4))
            env.agent_pos, env.agent_dir = (1,1), 2
            world.observation = world._serialize(env.gen_obs())
            before = world.observation
            self.assertEqual(describe_front(before)['object'], 'wall')
            self.assertFalse(describe_front(before)['forward_passable'])
            after, reward, terminated, truncated, _ = world.step('forward')
            self.assertTrue(after['last_transition']['blocked_forward'])
            self.assertFalse(after['last_transition']['moved'])
            agent = OllamaAgent()
            agent.learn(before, 'forward', reward, after, terminated, truncated)
            response = io.BytesIO(b'{"message":{"content":"ACTION: right"},"done":true}\n')
            with patch('babyai.ollama_agent.request.urlopen', return_value=response) as call:
                self.assertEqual(agent.choose_action(after), 'right')
                prompt = json.loads(call.call_args.args[0].data)['messages'][-1]['content']
                self.assertIn('Walls are impassable', prompt)
                self.assertIn('"blocked_forward": true', prompt)
                self.assertIn('"object": "wall"', prompt)
                self.assertIn('"recent_transitions": [{', prompt)
            turned, *_ = world.step('right')
            self.assertTrue(turned['last_transition']['turned'])
            self.assertFalse(turned['last_transition']['moved'])
            agent.end_episode()
            self.assertFalse(agent.recent_transitions)
            self.assertNotIn('last_transition', world.reset())
        finally:
            world.close()

    def test_thinking_is_not_translated_even_for_old_checkpoints(self):
        import io
        import json
        from unittest.mock import patch
        from babyai.ollama_agent import OllamaAgent
        agent = OllamaAgent()
        agent.translate_thinking = True  # Attribute from an older saved session.
        original = 'A red ball is ahead.'
        response = io.BytesIO(json.dumps({'message':{'thinking':original, 'content':'ACTION: forward'}, 'done':True}).encode())
        with patch('babyai.ollama_agent.request.urlopen', return_value=response) as call:
            self.assertEqual(agent.choose_action({}), 'forward')
            self.assertEqual(call.call_count, 1)
        self.assertEqual(agent.trace['thinking'], original)
        self.assertNotIn('thinking_ja', agent.trace)

    def test_session_retains_all_ollama_logs(self):
        import importlib.util
        if not importlib.util.find_spec('minigrid'):
            self.skipTest('MiniGrid is not installed')
        import io
        import cloudpickle
        from unittest.mock import patch
        from babyai.app import Session
        session = Session({'agent':'ollama', 'max_steps':1})
        try:
            with patch('babyai.ollama_agent.request.urlopen', side_effect=lambda *a, **kw: io.BytesIO(b'{"message":{"thinking":"test","content":"forward"},"done":true}\n')):
                for _ in range(25):
                    session.step()
            self.assertEqual(len(session.llm_history), 25)
            self.assertEqual(session.llm_history[0]['step'], 1)
            self.assertEqual([row['step'] for row in session.llm_history], [1]*25)
            self.assertEqual([row['episode'] for row in session.llm_history], list(range(1,26)))
            self.assertEqual([row['total_steps'] for row in session.llm_history], list(range(1,26)))
            self.assertEqual(session.snapshot()['llm_log_count'], 25)
            restored = cloudpickle.loads(cloudpickle.dumps(session))
            self.assertEqual(len(restored.llm_history), 25)
            restored.world.close()
        finally:
            session.world.close()

    def test_ollama_stream_progress_and_invalid_action(self):
        import io
        import json
        from unittest.mock import patch
        from babyai.ollama_agent import OllamaAgent
        agent = OllamaAgent()
        chunks = [
            {'message':{'thinking':'The door is ahead.'}, 'done':False},
            {'message':{'content':'tog'}, 'done':False},
            {'message':{'content':'gle'}, 'done':True}]
        stream = io.BytesIO(b'\n'.join(json.dumps(c).encode() for c in chunks))
        progress = []
        with patch('babyai.ollama_agent.request.urlopen', return_value=stream) as call:
            self.assertEqual(agent.choose_action({'mission':'open the door'}, progress.append), 'toggle')
            self.assertTrue(json.loads(call.call_args.args[0].data)['stream'])
        self.assertTrue(any(t['status']=='thinking' and t['thinking'] for t in progress))
        self.assertEqual(progress[-1]['action'], 'toggle')
        stream = io.BytesIO(b'{"message":{"content":"invalid"},"done":true}\n')
        with patch('babyai.ollama_agent.request.urlopen', return_value=stream):
            with self.assertRaises(ValueError):
                agent.choose_action({})
        self.assertEqual(agent.trace['status'], 'error')
        self.assertIsNone(agent.trace['action'])
        content = '正面のドアを開けます。\nACTION: toggle'
        stream = io.BytesIO(json.dumps({'message':{'content':content}, 'done':True}, ensure_ascii=False).encode('utf-8'))
        with patch('babyai.ollama_agent.request.urlopen', return_value=stream) as call:
            self.assertEqual(agent.choose_action({}), 'toggle')
            self.assertIn('Japanese', str(json.loads(call.call_args.args[0].data)['messages']))
        self.assertEqual(agent.trace['content'], content)
        content = '前進します。\n**action: forward**'
        stream = io.BytesIO(json.dumps({'message':{'content':content}, 'done':True}, ensure_ascii=False).encode('utf-8'))
        with patch('babyai.ollama_agent.request.urlopen', return_value=stream):
            self.assertEqual(agent.choose_action({}), 'forward')

    def test_real_parallel_ppo_and_resume(self):
        import importlib.util
        if not importlib.util.find_spec('minigrid'):
            self.skipTest('MiniGrid is not installed')
        import threading
        import cloudpickle
        from babyai.app import Session
        from babyai.parallel_training import train_parallel
        session = Session({'agent':'ppo', 'cpu_workers':2, 'max_steps':4, 'seed':42})
        try:
            train_parallel(session, 3, threading.Event())
            self.assertEqual(len(session.history), 3)
            self.assertEqual(session.total_steps, sum(r['steps'] for r in session.history))
            self.assertEqual(session.execution['workers'], 2)
            self.assertFalse(session.agent.buffer)
            restored = cloudpickle.loads(cloudpickle.dumps(session))
            try:
                restored.step()
                self.assertEqual(restored.steps, 1)
                self.assertEqual(len(restored.history), 3)
            finally:
                restored.world.close()
        finally:
            session.world.close()

    def test_observation_speed_changes_during_run(self):
        from unittest.mock import Mock, patch
        from babyai import app
        session = Mock(history=[])
        session.step.side_effect = lambda: session.history.append({})
        stop = Mock()
        stop.is_set.return_value = False
        def change_speed(delay):
            app.OBSERVATION_DELAY = .03
        stop.wait.side_effect = change_speed
        with patch.object(app, 'SESSION', session), patch.object(app, 'STOP', stop), patch.object(app, 'save_session'), patch.object(app, 'OBSERVATION_DELAY', .16):
            app.run(2, True)
            self.assertEqual([call.args[0] for call in stop.wait.call_args_list], [.16, .03])
            session.history.clear()
            stop.wait.reset_mock()
            app.run(2, False)
            stop.wait.assert_not_called()

    def test_session_save_resume(self):
        from unittest.mock import patch
        import cloudpickle
        from babyai import app
        class World:
            def __init__(self, **kwargs):
                self.counter=0
            def reset(self):
                self.counter=0
                return {'image':[[[1,0,0]]*7 for _ in range(7)],'direction':0,'mission':'test'}
            def state(self):
                return {'objects':[], 'position':[0,0], 'front':[1,0], 'carrying':None,'direction':0}
            def step(self, action):
                self.counter+=1
                return {'image':[[[1,0,0]]*7 for _ in range(7)],'direction':0,'mission':'test'},float(self.counter==3),self.counter==3,False,self.counter==3
            def close(self):
                pass
        with patch.object(app,'BabyAIEnvironment',World):
            session=app.Session({'agent':'ppo'})
        session.step()
        restored=cloudpickle.loads(cloudpickle.dumps(session))
        restored.step()
        restored.step()
        self.assertEqual(restored.history[0]['steps'],3)
        self.assertTrue(restored.history[0]['success'])
        restored.step()
        self.assertEqual(restored.steps,1)
        self.assertEqual(restored.total_steps,4)

    def test_http_page_and_state(self):
        import threading
        import json
        from urllib.request import urlopen
        from urllib.request import Request
        from urllib.error import HTTPError
        from unittest.mock import Mock, patch
        from http.server import ThreadingHTTPServer
        from babyai import app
        server=ThreadingHTTPServer(('127.0.0.1',0),app.Handler)
        worker=threading.Thread(target=server.serve_forever,daemon=True)
        worker.start()
        try:
            base=f'http://127.0.0.1:{server.server_port}'
            with urlopen(base+'/') as r:
                self.assertIn(b'/api/run',r.read())
            with urlopen(base+'/api/state') as r:
                self.assertIn('running',json.load(r))
            with urlopen(base+'/static/log_follow.js') as r:
                self.assertIn(b'BabyAILogFollow', r.read())
            from types import SimpleNamespace
            with patch.object(app, 'SESSION', SimpleNamespace(id='test', llm_history=[{'step':i} for i in range(25)])):
                with urlopen(base+'/api/llm-logs?after=20') as r:
                    data = json.load(r)
                    self.assertEqual(data['id'], 'test')
                    self.assertEqual([row['step'] for row in data['logs']], list(range(20,25)))
            with patch.object(app, 'WORKER', Mock(is_alive=lambda: True)), patch.object(app, 'OBSERVATION_DELAY', .16):
                req = Request(base+'/api/speed', data=b'{"speed_ms":80}', headers={'Content-Type':'application/json'})
                with urlopen(req) as r:
                    self.assertEqual(json.load(r)['speed_ms'], 80)
                self.assertEqual(app.OBSERVATION_DELAY, .08)
                req = Request(base+'/api/speed', data=b'{"speed_ms":-1}', headers={'Content-Type':'application/json'})
                with self.assertRaises(HTTPError) as error:
                    urlopen(req)
                self.assertEqual(error.exception.code, 400)
        finally:
            server.shutdown()
            server.server_close()
            worker.join()

if __name__ == '__main__':
    unittest.main()
