import hashlib
import json
from pathlib import Path
import tempfile
import sqlite3
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from service import SessionRequest, validate_rig, decode_image
from model_selection import resolve_model
from PIL import Image
import io


def rig():
    # Synthetic geometry, not a customer/demo character asset.
    return {'rig_id':'synthetic','joints':2,'joint_names':['root','child'],'parents':[-1,0],
            'rest_translations':[[0,0,3],[0,.3,0]],
            'rest_rotations_quat':[[0,0,0,1],[0,0,0,1]],'metric_scale':2,
            'coordinate_system':'opencv_right_handed_ydown_zforward_meters','units':'meters'}


class PortabilityTests(unittest.TestCase):
    def test_explicit_rig_and_single_root(self):
        with self.assertRaises(ValueError):SessionRequest.model_validate({})
        data=rig();self.assertEqual(validate_rig(data)['joints'],2)
        data['parents']=[-1,-1]
        with self.assertRaises(ValueError):validate_rig(data)

    def test_decode_uses_real_payload(self):
        out=io.BytesIO();Image.new('RGB',(40,25),(120,70,30)).save(out,format='PNG')
        rgb,ms=decode_image(out.getvalue())
        self.assertEqual(rgb.shape,(25,40,3));self.assertEqual(rgb[0,0].tolist(),[120,70,30])

    def test_relative_model_manifest_and_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'m').mkdir()
            (root/'m/config.yaml').write_text('model: example')
            (root/'m/runtime.pt').write_bytes(b'not-a-real-model')
            sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            spec={'id':'test','model_family':'CAST-L','backbone':'dinov3','global_step':1,'encoder_layers':16,
                  'decoder_layers':16,'q_dim':384,'num_heads':12,'image_size':256,
                  'cache_length':81,'amp_dtype':'bf16','source_sha256':'0'*64,
                  'checkpoint_name':'external.pt','verified':True,'config':'m/config.yaml',
                  'checkpoint':'m/runtime.pt','config_sha256':sha(root/'m/config.yaml'),
                  'checkpoint_sha256':sha(root/'m/runtime.pt')}
            (root/'active_model.json').write_text(json.dumps(spec))
            self.assertEqual(resolve_model(root,'dinov3')['checkpoint'],str(root/'m/runtime.pt'))
            (root/'m/runtime.pt').write_bytes(b'changed')
            with self.assertRaises(RuntimeError):resolve_model(root,'dinov3')
            spec['config']='../escape.yaml';(root/'active_model.json').write_text(json.dumps(spec))
            with self.assertRaises(ValueError):resolve_model(root,'dinov3')

    def test_admin_auth_no_model_required(self):
        import admin.app as mod
        secret='test-only-'+'a'*32
        with patch.object(mod,'PASSWORD',secret), patch.object(mod,'USER','testadmin'), \
                patch.object(mod.Sampler,'sample'), TestClient(mod.app) as client:
            self.assertEqual(client.get('/').status_code,200)
            self.assertEqual(client.get('/api/overview').status_code,401)
            self.assertEqual(client.post('/api/login',json={'username':'x','password':'bad'}).status_code,401)
            self.assertEqual(client.post('/api/login',json={'username':'testadmin','password':secret}).status_code,200)
            self.assertEqual(client.get('/api/live-history').status_code,200)
            self.assertEqual(client.get('/api/logs',params={'source':'../../.env'}).status_code,422)
            self.assertEqual(client.get('/assets/.env').status_code,404)
            self.assertEqual(client.post('/api/logout').status_code,200)
            self.assertEqual(client.get('/api/overview').status_code,401)

    def test_admin_database_connections_close_and_backup_commits(self):
        import admin.app as mod
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(mod, 'DB', root/'monitor.sqlite3'), \
                    patch.object(mod, 'LOCAL', root), \
                    patch.object(mod, 'BACKUP', root/'published.sqlite3'):
                mod.initialize()
                mod.event('info', 'test', 'saved event')
                with mod.connect() as connection:
                    self.assertEqual(connection.execute('SELECT count(*) FROM events').fetchone()[0], 1)
                with self.assertRaises(sqlite3.ProgrammingError):
                    connection.execute('SELECT 1')
                mod.Sampler.backup()
                # A live backup connection would keep this file locked on Windows.
                backup = root/'backup.sqlite3'
                backup.rename(root/'moved.sqlite3')
