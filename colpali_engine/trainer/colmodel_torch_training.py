import os

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import all_gather_tensor_autograd
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

from colpali_engine.collators import VisualRetrieverCollator
from colpali_engine.trainer.colmodel_training import ColModelTrainingConfig
from colpali_engine.utils.gpu_stats import print_gpu_utilization


class ColModelTorchTraining:
    """
    Class that contains the training and evaluation logic for a ColVision model.
    """

    def __init__(self, config: ColModelTrainingConfig) -> None:
        self.config = config
        self.model = self.config.model
        self.current_git_hash = os.popen("git rev-parse HEAD").read().strip()
        self.train_dataset = self.config.train_dataset
        self.eval_dataset = self.config.eval_dataset
        self.collator = VisualRetrieverCollator(
            processor=self.config.processor,
            max_length=self.config.max_length,
        )

        # Initialize distributed if needed
        if dist.is_available() and not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
            print("Distributed process group initialized.")
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        print(f"Local rank: {self.local_rank}, World size: {self.world_size}")

        device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(device)
        self.model.to(device)

        # Gradient checkpointing if supported
        if getattr(self.config.tr_args, "gradient_checkpointing", False):
            # huggingface models expose this
            try:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=self.config.tr_args.gradient_checkpointing_kwargs
                )
                if self._is_rank0():
                    print("Gradient checkpointing enabled.")
            except Exception as e:
                if self._is_rank0():
                    print("Warning: gradient_checkpointing_enable() not supported by model.")
                    print(e)

        self.model = DistributedDataParallel(self.model, device_ids=[self.local_rank], output_device=self.local_rank)

        self.model = torch.compile(
            self.model,
            backend="inductor",
            dynamic=True,  # or True if you know shapes will vary a lot
        )

    def _is_rank0(self) -> bool:
        return not dist.is_initialized() or dist.get_rank() == 0

    def train(self) -> None:
        # Mixed precision setup
        use_amp = getattr(self.config, "use_amp", False)
        scaler = torch.amp.GradScaler("cuda") if use_amp else None
        max_grad_norm = getattr(self.config.tr_args, "max_grad_norm", None)
        print(f"Using AMP: {use_amp}, Max grad norm: {max_grad_norm}")

        sampler = DistributedSampler(self.train_dataset) if dist.is_initialized() else None
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.tr_args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.collator,
            num_workers=self.config.tr_args.dataloader_num_workers,
            prefetch_factor=2,
            pin_memory=True,
            drop_last=True,
        )

        # Evaluation loader
        eval_loader = None
        if self.config.eval_dataset_loader is not None:
            eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=self.config.tr_args.per_device_eval_batch_size,
                collate_fn=self.collator,
            )
        elif self._is_rank0():
            print("No eval dataset provided. Skipping evaluation.")

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.tr_args.learning_rate,
            weight_decay=self.config.tr_args.weight_decay,
        )
        num_training_steps = self.config.tr_args.num_train_epochs * len(train_loader)
        warmup_steps = self.config.tr_args.warmup_steps

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
            return max(0.1, 1.0 - (1.0 - 0.1) * progress)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        loss_fn = self.config.loss_func

        def gather_with_grad(x: torch.Tensor) -> torch.Tensor:
            return all_gather_tensor_autograd(x, gather_dim=0, group=dist.group.WORLD)

        # Training loop
        # only rank0 should display
        if self._is_rank0():
            pbar = tqdm(total=num_training_steps, desc="Training", leave=True)
        else:
            pbar = None

        for epoch in range(self.config.tr_args.num_train_epochs):
            if sampler:
                sampler.set_epoch(epoch)
            self.model.train()

            for step, batch in enumerate(train_loader):
                # Move batch to device
                batch = {k: v.to(self.model.device, non_blocking=True) for k, v in batch.items()}

                # Forward with optional AMP
                with torch.amp.autocast("cuda", enabled=use_amp):
                    q_embed = self.model(
                        input_ids=batch["query_input_ids"], attention_mask=batch["query_attention_mask"]
                    )
                    d_embed = self.model(**{k[4:]: v for k, v in batch.items() if k.startswith("doc_")})
                    neg_embed = None
                    if "neg_doc_input_ids" in batch:
                        neg_embed = self.model(**{k[8:]: v for k, v in batch.items() if k.startswith("neg_doc")})

                    def pad_to_max_len_right(x: torch.Tensor) -> torch.Tensor:
                        """
                        Right-pad x along dim=1 so that all ranks share the same length.

                        Args:
                            x: Tensor of shape [B, L, D] (or [B, L] if 2D)
                        Returns:
                            Padded tensor of shape [B, max_L, D], with zeros on the right.
                        """
                        # 1) local length
                        local_len = x.size(1)
                        # 2) get global max length
                        len_tensor = torch.tensor(local_len, device=x.device)
                        dist.all_reduce(len_tensor, op=dist.ReduceOp.MAX)
                        max_len = len_tensor.item()

                        # 3) if shorter, pad on the right of dim=1
                        if local_len < max_len:
                            pad_amount = max_len - local_len
                            # torch.nn.functional.pad takes (D_left, D_right, L_left, L_right)
                            x = torch.nn.functional.pad(x, (0, 0, 0, pad_amount), value=0.0)
                        return x

                    # Usage before gathering:
                    d_embed = pad_to_max_len_right(d_embed)
                    if neg_embed is not None:
                        neg_embed = pad_to_max_len_right(neg_embed)

                    # Now safe to all_gather:
                    d_global = gather_with_grad(d_embed)
                    n_global = gather_with_grad(neg_embed) if neg_embed is not None else None

                    # loss = loss_fn(q_global, d_global) if n_global is None else loss_fn(q_global, d_global, n_global)
                    loss = (
                        loss_fn(q_embed, d_global, offset=(dist.get_rank() * batch["query_input_ids"].shape[0]))
                        if n_global is None
                        else loss_fn(
                            q_embed, d_global, n_global, offset=(dist.get_rank() * batch["query_input_ids"].shape[0])
                        )
                    )

                # Backward
                if use_amp:
                    scaler.scale(loss).backward()
                    if max_grad_norm:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

                if self._is_rank0() and not isinstance(train_loader, DataLoader):
                    # advance the global bar
                    # you can also show epoch/step in the postfix if you like:
                    pbar.set_postfix(epoch=epoch + 1, step=step + 1, refresh=False)
                    pbar.update(1)

                if self._is_rank0() and step % 10 == 0:
                    print(f"Step {step}/{len(train_loader)}")
                    print(f"Query embedding shape: {q_embed.shape}")
                    print(f"Document embedding shape: {d_embed.shape}")
                    if neg_embed is not None:
                        print(f"Negative document embedding shape: {neg_embed.shape}")

                    # print(f"Gathered query embedding shape: {q_global.shape}")
                    print(f"Gathered document embedding shape: {d_global.shape}")
                    if neg_embed is not None:
                        print(f"Gathered negative document embedding shape: {n_global.shape}")

                    print(f"Batch size: {batch['query_input_ids'].shape[0]}")

                    print_gpu_utilization()
                    print(f"Local loss: {loss.item()}")
                    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
                    print(f"Epoch: {epoch + 1}/{self.config.tr_args.num_train_epochs}")
                    print(f"World size: {dist.get_world_size()}")
                    # with torch.no_grad():
                    #     avg_loss = loss.detach()
                    #     dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    #     avg_loss /= dist.get_world_size()
                    #     print(f"Local loss: {avg_loss.item()}")

            # Optional evaluation
            if eval_loader and self._is_rank0():
                self.evaluate(eval_loader)

        self.model = self.model.module if hasattr(self.model, "module") else self.model
        # Final actions
        if self._is_rank0():
            pbar.close()
            print("Training complete. Saving model.")
            self.save()
            print("Model saved.")

        if dist.is_initialized():
            dist.destroy_process_group()

    def eval(self) -> None:
        raise NotImplementedError("Evaluation is not implemented yet.")

    def save(self):
        """
        Save the model with its training config, as well as the tokenizer and processor if provided.
        """
        self.model.save_pretrained(self.config.output_dir)
        self.config.processor.save_pretrained(self.config.output_dir)

        # Save git hash of the commit at beginning of training
        with open(f"{self.config.output_dir}/git_hash.txt", "w") as f:
            f.write(self.current_git_hash)
