"""Model selection and checkpoint preparation without external weights."""
from contextlib import redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

from model_selection import MODEL_CONFIGS, load_model_config, resolve_model
from tools import prepare_checkpoint
from utils.config_utils import instantiate_from_config


class SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(2, 2, bias=False)
        self.decoder = torch.nn.Linear(2, 2, bias=False)
        self.grace = torch.nn.Linear(2, 2, bias=False)

    def trainable_state_dict(self):
        return self.state_dict()

    def load_trainable_state_dict(self, state):
        self.load_state_dict(state, strict=True)


class ModelTests(unittest.TestCase):
    def test_both_model_configs_construct_with_canonical_names(self):
        self.assertEqual(set(MODEL_CONFIGS), {'CAST-B', 'CAST-L'})
        for family in MODEL_CONFIGS:
            with self.subTest(model=family):
                config = load_model_config(family)
                # The actual model is constructed on meta; no DINO download or allocation.
                config['model']['params']['image_backbone_cfg'] = {'target': 'torch.nn.Identity'}
                with torch.device('meta'):
                    model = instantiate_from_config(config['model'])
                keys = model.trainable_state_dict()
                for prefix in ('encoder.', 'decoder.', 'grace.'):
                    self.assertTrue(any(key.startswith(prefix) for key in keys))

    def test_unknown_model_family_rejected(self):
        with self.assertRaises(ValueError):
            load_model_config('unregistered-model')

    def prepare(self, root, family, state):
        checkpoint = root/'source.pt'
        torch.save({'model_state': state, 'global_step': 12}, checkpoint)
        arguments = ['prepare_checkpoint.py', '--model', family,
                     '--checkpoint', str(checkpoint), '--model-id', 'test-model', '--activate']
        with patch.object(prepare_checkpoint, 'SETTINGS', SimpleNamespace(assets_dir=root/'assets')), \
                patch.object(prepare_checkpoint, 'instantiate_from_config', return_value=SmallModel()) as factory, \
                patch('sys.argv', arguments), redirect_stdout(io.StringIO()):
            prepare_checkpoint.main()
        self.assertEqual(factory.call_args.args[0]['target'], 'models.base_model.model.BaseModel')
        return root/'assets'

    def test_checkpoint_values_and_family_survive_preparation(self):
        for family in MODEL_CONFIGS:
            with self.subTest(model=family), tempfile.TemporaryDirectory() as temp:
                state = SmallModel().state_dict()
                assets = self.prepare(Path(temp), family, state)
                selected = resolve_model(assets, 'dinov3')
                self.assertEqual(selected['public']['model_family'], family)
                saved = torch.load(selected['checkpoint'], weights_only=True)['model_state']
                self.assertEqual(set(saved), set(state))
                for key in state:
                    self.assertTrue(torch.equal(saved[key], state[key]))
                config = yaml.safe_load(Path(selected['config']).read_text())
                self.assertEqual(config['model_family'], family)
                backbone = config['model']['params']['image_backbone_cfg']['params']
                self.assertEqual(backbone['weights'], '${CAST_DINOV3_WEIGHTS}')

    def test_invalid_checkpoint_cannot_activate(self):
        state = SmallModel().state_dict()
        invalid_states = [
            {key: value for key, value in state.items() if key != 'encoder.weight'},
            {**state, 'unexpected.weight': torch.ones(2, 2)},
            {**state, 'decoder.weight': torch.ones(3, 3)},
        ]
        for bad_state in invalid_states:
            with self.subTest(keys=list(bad_state)), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                with self.assertRaises(RuntimeError):
                    self.prepare(root, 'CAST-B', bad_state)
                self.assertFalse((root/'assets/active_model.json').exists())
