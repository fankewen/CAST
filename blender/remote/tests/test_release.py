import unittest
from pathlib import Path
from tools.build_release import ROOT, candidate_files, check_text
from client.pipeline_client import PipelinedMotionClient


class ReleaseSafetyTests(unittest.TestCase):
    def test_release_includes_current_runtime_once(self):
        paths = {p.relative_to(ROOT).as_posix() for p in candidate_files()}
        self.assertIn('inference/deploy_realtime_video2motion.py', paths)
        self.assertIn('utils/motion_io.py', paths)
        self.assertIn('visualization/motion_visualization.py', paths)
        self.assertIn('blender/rig_adapter.py', paths)
        self.assertIn('blender/remote/client/pipeline_client.py', paths)
        self.assertFalse(any('cast_core' in Path(p).parts for p in paths))
        self.assertNotIn('blender/remote/pipeline_client.py', paths)
        self.assertNotIn('blender/remote/client/rig_adapter.py', paths)
        self.assertFalse(any('var' in Path(p).parts or '.env' == Path(p).name for p in paths))

    def test_empty_secret_template_is_safe(self):
        check_text('.env.example', b'CAST_API_KEY=\nCAST_ADMIN_PASSWORD=\nCAST_GPU=0\n')

    def test_literal_secret_rejected_without_echo(self):
        with self.assertRaises(ValueError) as exc:
            check_text('example', b'CAST_API_KEY=not-a-real-test-secret\n')
        self.assertNotIn('not-a-real-test-secret', str(exc.exception))

    def test_private_tokens_and_binary_rejected(self):
        with self.assertRaises(ValueError): check_text('x.py', b'opaque-test-value', ['opaque-test-value'])
        with self.assertRaises(ValueError): check_text('x.py', b'hi\0there')

    def test_loopback_and_package_versions(self):
        check_text('x', b'http://127.0.0.1:8080; opencv-python-headless==4.11.0.86')

    def test_client_requires_rig_before_network(self):
        c = PipelinedMotionClient('http://127.0.0.1:1', 'test-only-key', window=10)
        try:
            with self.assertRaises(ValueError): c.start(None)
        finally: c.close()
