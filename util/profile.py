from torch import nn

class Profiler(nn.Module):
    def __init__(self, model):
        super(Profiler, self).__init__()
        self.model = model
        self.hooks = []
        self.macs = []
        self.params = []

        def hook_conv(module, input, output):
            self.macs.append(output.size(1) * output.size(2) * output.size(3) *
                             module.weight.size(-1) * module.weight.size(-1) * input[0].size(1) / module.groups)
            self.params.append(module.weight.size(0) * module.weight.size(1) *
                               module.weight.size(2) * module.weight.size(3) + module.weight.size(1))

        def hook_gelu(module, input, output):
            if len(output[0].size()) > 3:
                self.macs.append(output.size(1) * output.size(2) * output.size(3))
            else:
                self.macs.append(output.size(1) * output.size(2))

        def hook_avgpool(module, input, output):
            self.macs.append(output.size(1) * output.size(2) * output.size(3) * module.kernel_size * module.kernel_size)

        for _, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.hooks.append(module.register_forward_hook(hook_conv))
            elif isinstance(module, nn.GELU):
                self.hooks.append(module.register_forward_hook(hook_gelu))
            elif isinstance(module, nn.AvgPool2d):
                self.hooks.append(module.register_forward_hook(hook_avgpool))

    def forward(self, x):
        self.model.to(x.device)
        _ = self.model(x)
        for handle in self.hooks:
            handle.remove()
        return self.macs, self.params