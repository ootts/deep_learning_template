#pragma once
#include <torch/extension.h>


at::Tensor SigmoidFocalLoss_forward_cuda(
		const at::Tensor& logits,
                const at::Tensor& targets,
		const int num_classes, 
		const float gamma, 
		const float alpha); 

at::Tensor SigmoidFocalLoss_backward_cuda(
			     const at::Tensor& logits,
                             const at::Tensor& targets,
			     const at::Tensor& d_losses,
			     const int num_classes,
			     const float gamma,
			     const float alpha);

