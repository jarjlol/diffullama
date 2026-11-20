import pathlib,tempfile,unittest
import operator_start_guard as guard
class GuardTests(unittest.TestCase):
 def test_active_consumers(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=pathlib.Path(tmp)
   def add(pid,args):
    p=root/str(pid);p.mkdir();(p/'cmdline').write_bytes(('\x00'.join(args)+'\x00').encode())
   add(101,['bash','./lane.sh','0','dream-base','dream'])
   add(102,['bash','./queue.sh','0'])
   add(103,['bash','./queue.sh','1'])
   add(104,['bash','unrelated.sh','0'])
   self.assertEqual({r['pid'] for r in guard.existing('lane',0,root)}, {101,102})
   self.assertEqual({r['pid'] for r in guard.existing('queue',0,root)}, {102})
   self.assertEqual({r['pid'] for r in guard.existing('lane',1,root)}, {103})
   self.assertEqual(guard.existing('queue',0,root,exclude=(102,)),[])
if __name__=='__main__':unittest.main()
