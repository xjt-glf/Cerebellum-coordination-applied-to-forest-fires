"""Joint planning must not silently fall back to heuristic UAV decisions."""
import threading
import unittest
from types import SimpleNamespace
from fastapi import HTTPException
from webapp import server

class JointAgentContractTests(unittest.TestCase):
    def setUp(self):
        self.sid = 'test_joint'
        self.controller = SimpleNamespace(normalize=lambda orders: orders,
                                          env=SimpleNamespace(grid_size=(100,100)))
        server._SESS[self.sid] = dict(status='waiting_personnel', brain_kind='agent',
            n_uavs=4, personnel_controller=self.controller, personnel_request={'t':20},
            personnel_submissions=[], wake=threading.Event())

    def tearDown(self):
        server._SESS.pop(self.sid, None)

    def test_joint_submission_requires_all_uav_orders(self):
        with self.assertRaises(HTTPException) as ctx:
            server.api_personnel_command(self.sid, server.PersonnelIn(orders=[]))
        self.assertEqual(ctx.exception.status_code,422)
        self.assertFalse(server._SESS[self.sid]['wake'].is_set())

    def test_nonfinite_uav_target_rejected_atomically(self):
        orders=[{'task':'explore','target':[10,20]} for _ in range(4)]
        orders[2]['target'][0]=float('nan')
        with self.assertRaises(HTTPException):
            server.api_personnel_command(self.sid, server.PersonnelIn(uavs=orders))
        self.assertEqual(server._SESS[self.sid]['personnel_submissions'],[])

    def test_complete_joint_plan_queues_and_wakes(self):
        orders=[{'task':'explore','target':[10,20]} for _ in range(4)]
        answer=server.api_personnel_command(self.sid, server.PersonnelIn(
            uavs=orders, orders=[{'person_ids':[1],'task':'move','target':[12,50]}],
            expected_tick=20, source='codex_joint'))
        queued=server._SESS[self.sid]['personnel_submissions']
        self.assertTrue(answer['ok'])
        self.assertEqual(queued[0]['uavs'],orders)
        self.assertTrue(server._SESS[self.sid]['wake'].is_set())
        with self.assertRaises(HTTPException) as ctx:
            server.api_personnel_command(self.sid, server.PersonnelIn(uavs=orders))
        self.assertEqual(ctx.exception.status_code,409)
