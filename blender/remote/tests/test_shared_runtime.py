"""Shared imports and restricted remote checkpoint loading, without a GPU."""
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
from inference import deploy_realtime_video2motion as deploy
from settings import REPOSITORY_ROOT


class NonTensorMetadata:
    pass


class SharedRuntimeTests(unittest.TestCase):
    def test_local_engine_forwards_sam_selection(self):
        from blender.cast_engine_server import dispatch
        engine = MagicMock()
        engine.load_weights.return_value = {'ok': True}
        request = {'cmd': 'load_weights', 'config': 'cast.yaml', 'checkpoint': 'cast.pt',
                   'sam_cfg': 'configs/sam2.1/custom.yaml', 'sam_ckpt': 'custom-sam.pt'}
        self.assertEqual(dispatch(engine, request), {'ok': True})
        engine.load_weights.assert_called_once_with(
            'cast.yaml', 'cast.pt', sam_cfg=request['sam_cfg'], sam_ckpt=request['sam_ckpt'],
        )

    def test_model_and_deployment_imports_use_current_repository(self):
        from models.base_model import model
        from utils import motion_io
        from visualization import motion_visualization
        for module, relative in (
            (deploy, 'inference/deploy_realtime_video2motion.py'),
            (model, 'models/base_model/model.py'),
            (motion_io, 'utils/motion_io.py'),
            (motion_visualization, 'visualization/motion_visualization.py'),
        ):
            self.assertEqual(Path(module.__file__).resolve(), REPOSITORY_ROOT/relative)

    def test_remote_rejects_pickled_objects_but_local_compatibility_is_preserved(self):
        config = {'data': {'image_size': 16, 'max_joints': 2}, 'model': {}}
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp)/'checkpoint.pt'
            torch.save({'model_state': {}, 'metadata': NonTensorMetadata()}, checkpoint)
            model = MagicMock()
            model.grace = None
            with patch.object(deploy, 'load_yaml_config', return_value=config), \
                    patch.object(deploy, 'instantiate_from_config', return_value=model):
                with self.assertRaises(pickle.UnpicklingError):
                    deploy.RealtimeVideo2Motion(
                        config='unused', checkpoint=str(checkpoint),
                        compile_cache_dir=temp, checkpoint_weights_only=True,
                    )
                model.load_trainable_state_dict.assert_not_called()
                deploy.RealtimeVideo2Motion(
                    config='unused', checkpoint=str(checkpoint), compile_cache_dir=temp,
                )
                model.load_trainable_state_dict.assert_called_once_with({})

    def test_remote_loads_tensor_checkpoint(self):
        config = {'data': {'image_size': 16, 'max_joints': 2}, 'model': {}}
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp)/'checkpoint.pt'
            state = {'weight': torch.arange(4)}
            torch.save({'model_state': state}, checkpoint)
            model = MagicMock()
            model.grace = None
            with patch.object(deploy, 'load_yaml_config', return_value=config), \
                    patch.object(deploy, 'instantiate_from_config', return_value=model):
                deploy.RealtimeVideo2Motion(
                    config='unused', checkpoint=str(checkpoint), compile_cache_dir=temp,
                    checkpoint_weights_only=True,
                )
            loaded = model.load_trainable_state_dict.call_args.args[0]
            self.assertTrue(torch.equal(loaded['weight'], state['weight']))
