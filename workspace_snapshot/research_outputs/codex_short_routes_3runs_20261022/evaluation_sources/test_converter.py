import unittest,sys,copy
from types import SimpleNamespace as NS
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from webapp.route_converter import ShortRouteConverter,validate_route
from lowlevel.task_spec import TeamPlan,UavOrder
class ConverterTest(unittest.TestCase):
    def make(self,enabled=True):
        env=NS(t=0,orders=[UavOrder()],env=NS(grid_size=(100,100),uavs=[NS(x=10,y=10)]))
        def set_plan(orders):env.orders=copy.deepcopy(orders);return 'obs'
        env.set_plan=set_plan
        goals=NS(current=env.orders,manager=NS(sync=lambda *a:None))
        def apply(plan,source=None):
            orders=TeamPlan.from_llm_json(plan,1).orders;env.set_plan(orders);return 'obs',{'uavs':[o.to_dict() for o in orders]},[]
        goals.apply=apply
        return env,ShortRouteConverter(env,goals,enabled=enabled)
    def test_bounds_and_straight_contract(self):
        with self.assertRaises(ValueError):validate_route([[40,10],[40,20]],(10,10))
        with self.assertRaises(ValueError):validate_route([[15,10],[20,10]],(10,10))
        with self.assertRaises(ValueError):validate_route([[0,10],[10,15]],(10,10))
        self.assertEqual(validate_route([[16,10],[16,16]],(10,10))[1],12)
    def test_switches_immediately_and_holds_terminal(self):
        e,c=self.make();c.apply({'uavs':[{'task':'explore','route':[[16,10],[16,16]]}]})
        e.t=4;e.env.uavs[0].x=14;c.tick();self.assertEqual(e.orders[0].target,(16.,16.))
        e.t=10;e.env.uavs[0].x=16;e.env.uavs[0].y=15;c.tick();self.assertTrue(c.routes[0]['done']);self.assertEqual(e.orders[0].target,(16.,16.))
    def test_charge_preemption_never_resurrects(self):
        e,c=self.make();c.apply({'uavs':[{'task':'explore','route':[[16,10],[16,16]]}]});e.orders=[UavOrder(task=2)];c.tick();e.env.uavs[0].x=16;c.tick();self.assertEqual(e.orders[0].task,2);self.assertTrue(c.routes[0]['done'])
    def test_off_does_not_advance(self):
        e,c=self.make(False);c.apply({'uavs':[{'task':'explore','route':[[16,10],[16,16]]}]});e.env.uavs[0].x=16;c.tick();self.assertEqual(e.orders[0].target,(16.,10.))
if __name__=='__main__':unittest.main()
