
class Solver(object):
    def __init__(self,
                 dataloader,
                 dataloader_val,
                 model,
                 output_fn,
                 train_loss_fn,
                 eval_metric_fns,
                 optim):
        self._dataloader = dataloader
        self._dataloader_val = dataloader_val
        self.model = model
        self.model_params = [p for p in model.parameters() if p.requires_grad]
        self._output_fn = output_fn
        self._train_loss_fn = train_loss_fn
        self._eval_metric_fns = eval_metric_fns
        self.optim = optim

        self._before_run = []
        self._before_train = []
        self._before_step = []
        self._after_train_batch = []
        self._before_eval = []
        self._after_eval_batch = []
        self._after_eval = []

    def run(self, max_epoch=100):
        _ = [callback(self) for callback in self._before_run]

        for epoch in range(max_epoch):
            self.epoch = epoch

            _ = [callback(self) for callback in self._before_train]
            self.training = True
            self.model.train()

            for i, datum in enumerate(self._dataloader):
                self.batch = i
                self.datum = datum
                self.output = self._output_fn(self)
                self.train_loss = self._train_loss_fn(self)
                self.eval_metric = [f(self) for f in self._eval_metric_fns]
                self.optim.zero_grad()
                self.train_loss.backward()
                assert all(
                        (p.grad.data != p.grad.data).sum() == 0
                        for p in self.model_params if p.grad is not None)
                #assert not any(
                #        p.grad.data.abs().max() > 1e+8
                #        for p in self.model_params if p.grad is not None)
                _ = [callback(self) for callback in self._before_step]
                self.optim.step()
                _ = [callback(self) for callback in self._after_train_batch]

            _ = [callback(self) for callback in self._before_eval]
            self.training = False
            self.model.eval()
            for i, datum in enumerate(self._dataloader_val):
                self.batch = i
                self.datum = datum
                self.output = self._output_fn(self)
                self.valid_loss = self._train_loss_fn(self)
                self.eval_metric = [f(self) for f in self._eval_metric_fns]
                _ = [callback(self) for callback in self._after_eval_batch]
            _ = [callback(self) for callback in self._after_eval]

    def register_callback(self, key, func):
        getattr(self, '_' + key).append(func)
