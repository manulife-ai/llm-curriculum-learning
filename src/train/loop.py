import contextlib

import torch


def run_train_step(
    model,
    batch,
    optimizer,
    lr_scheduler,
    mixer,
    schedule,
    global_step: int,
    grad_accumulation_steps: int,
    max_grad_norm: float,
    precision: str,
    device: str,
):
    p_ar = schedule.get_p_ar(global_step)
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if precision == "bf16" and device.startswith("cuda") and torch.cuda.is_available()
        else contextlib.nullcontext()
    )
    with autocast_context:
        mixed_input_ids, mix_stats = mixer.mix(model, batch, p_ar)
        outputs = model(
            input_ids=mixed_input_ids,
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss / grad_accumulation_steps

    loss.backward()
    step_completed = False
    if (global_step + 1) % grad_accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step_completed = True

    return {
        "loss": float(outputs.loss.item()),
        "p_ar": float(p_ar),
        "replacement_rate": float(mix_stats["replacement_rate"]),
        "step_completed": step_completed,
    }
