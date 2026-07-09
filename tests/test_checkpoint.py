from __future__ import annotations

import torch
import torch.nn as nn

from bharat.training.checkpointing import (
    load_checkpoint,
    save_checkpoint,
)


class TestCheckpointResume:
    def test_optimizer_state_restored(self, tmp_path):
        model = nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        # Train one step
        x = torch.randn(5, 10)
        y = torch.randn(5, 10)
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        optimizer.step()

        saved_state = model.state_dict()
        saved_opt_state = optimizer.state_dict()

        # Save checkpoint
        ckpt_path = tmp_path / "resume_test.pt"
        save_checkpoint(ckpt_path, model, optimizer=optimizer, step=1)

        # Create new model and optimizer
        new_model = nn.Linear(10, 10)
        new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.01, momentum=0.9)

        # Load checkpoint
        result = load_checkpoint(
            ckpt_path, new_model, optimizer=new_optimizer, device="cpu", strict=False
        )

        # Verify step
        assert result["step"] == 1

        # Verify model weights restored
        for k in saved_state:
            assert torch.equal(
                new_model.state_dict()[k], saved_state[k]
            ), f"Model parameter {k} should match saved state"

        # Verify optimizer state restored
        new_state = new_optimizer.state_dict()
        for k in saved_opt_state:
            if k == "state":
                for param_id in saved_opt_state["state"]:
                    assert (
                        param_id in new_state["state"]
                    ), f"Optimizer state for param {param_id} should be restored"
                    for sk in saved_opt_state["state"][param_id]:
                        if isinstance(saved_opt_state["state"][param_id][sk], torch.Tensor):
                            assert torch.equal(
                                new_state["state"][param_id][sk],
                                saved_opt_state["state"][param_id][sk],
                            ), f"Optimizer {sk} for param {param_id} should match"

    def test_resume_after_interruption(self, tmp_path):
        """Train a tiny model for several iterations, save, resume, and verify
        the resumed run continues from the correct iteration."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

        # Train for 3 iterations (AdamW stores momentum/velocity state)
        for _it in range(3):
            x = torch.randn(5, 10)
            y = torch.randn(5, 10)
            loss = nn.MSELoss()(model(x), y)
            loss.backward()
            optimizer.step()

        final_iter = 3
        saved_params = {k: v.clone() for k, v in model.named_parameters()}
        ckpt_path = tmp_path / "interrupt_test.pt"
        save_checkpoint(ckpt_path, model, optimizer=optimizer, step=final_iter)

        # Simulate interruption: create fresh model and optimizer
        fresh_model = nn.Linear(10, 10)
        fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=0.01)

        # Resume
        result = load_checkpoint(
            ckpt_path, fresh_model, optimizer=fresh_optimizer, device="cpu", strict=False
        )

        resumed_step = result["step"]
        assert (
            resumed_step == final_iter
        ), f"Resumed step should be {final_iter}, got {resumed_step}"

        # Verify model weights were restored correctly
        for k in saved_params:
            assert torch.equal(fresh_model.state_dict()[k], saved_params[k])

        # Verify optimizer state was restored (AdamW has momentum/velocity per param)
        opt_state = fresh_optimizer.state_dict()
        assert (
            len(opt_state["state"]) > 0
        ), "Optimizer state dict should contain per-parameter state after loading"

        # Train one more step on the resumed model
        x = torch.randn(5, 10)
        y = torch.randn(5, 10)
        loss2 = nn.MSELoss()(fresh_model(x), y)
        loss2.backward()
        fresh_optimizer.step()

        # Verify the model continued training (weights changed from saved state)
        for k in saved_params:
            assert not torch.equal(
                fresh_model.state_dict()[k], saved_params[k]
            ), f"Parameter {k} should have changed after training one more step"
