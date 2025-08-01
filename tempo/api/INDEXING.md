>>> import torch
>>> torch.ones((5,3,2), requires_grad=True)
tensor([[[1., 1.],
         [1., 1.],
         [1., 1.]],

        [[1., 1.],
         [1., 1.],
         [1., 1.]],

        [[1., 1.],
         [1., 1.],
         [1., 1.]],

        [[1., 1.],
         [1., 1.],
         [1., 1.]],

        [[1., 1.],
         [1., 1.],
         [1., 1.]]], requires_grad=True)
>>> a = torch.ones((5,3,2), requires_grad=True)
>>> a[1]
tensor([[1., 1.],
        [1., 1.],
        [1., 1.]], grad_fn=<SelectBackward0>)
>>> a[1:]
tensor([[[1., 1.],
         [1., 1.],
         [1., 1.]],

        [[1., 1.],
         [1., 1.],
         [1., 1.]],

        [[1., 1.],
         [1., 1.],
         [1., 1.]],

        [[1., 1.],
         [1., 1.],
         [1., 1.]]], grad_fn=<SliceBackward0>)
>>> a[...,:1]
tensor([[[1.],
         [1.],
         [1.]],

        [[1.],
         [1.],
         [1.]],

        [[1.],
         [1.],
         [1.]],

        [[1.],
         [1.],
         [1.]],

        [[1.],
         [1.],
         [1.]]], grad_fn=<SliceBackward0>)
>>> a[torch.tensor([False,True,False,True, True])]
tensor([[[1., 1.],
         [1., 1.],
         [1., 1.]],

        [[1., 1.],
         [1., 1.],
         [1., 1.]],

        [[1., 1.],
         [1., 1.],
         [1., 1.]]], grad_fn=<IndexBackward0>)
>>> a[torch.tensor([False,True,False,True, True])]
KeyboardInterrupt
>>> a[...]
tensor([[[1., 1.],
         [1., 1.],
         [1., 1.]],

        [[1., 1.],
         [1., 1.],
         [1., 1.]],

        [[1., 1.],
         [1., 1.],
         [1., 1.]],

        [[1., 1.],
         [1., 1.],
         [1., 1.]],

        [[1., 1.],
         [1., 1.],
         [1., 1.]]], grad_fn=<AliasBackward0>)
>>> a[...,None]
tensor([[[[1.],
          [1.]],

         [[1.],
          [1.]],

         [[1.],
          [1.]]],


        [[[1.],
          [1.]],

         [[1.],
          [1.]],

         [[1.],
          [1.]]],


        [[[1.],
          [1.]],

         [[1.],
          [1.]],

         [[1.],
          [1.]]],


        [[[1.],
          [1.]],

         [[1.],
          [1.]],

         [[1.],
          [1.]]],


        [[[1.],
          [1.]],

         [[1.],
          [1.]],

         [[1.],
          [1.]]]], grad_fn=<UnsqueezeBackward0>)
>>> a[torch.tensor(1,3,4)]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: tensor() takes 1 positional argument but 3 were given
>>> a[torch.tensor([1,3,4])]
tensor([[[1., 1.],
         [1., 1.],
         [1., 1.]],

        [[1., 1.],
         [1., 1.],
         [1., 1.]],

        [[1., 1.],
         [1., 1.],
         [1., 1.]]], grad_fn=<IndexBackward0>)
>>> a[::2]
tensor([[[1., 1.],
         [1., 1.],
         [1., 1.]],

        [[1., 1.],
         [1., 1.],
         [1., 1.]],

        [[1., 1.],
         [1., 1.],
         [1., 1.]]], grad_fn=<SliceBackward0>)
>>>

------------------------------------------------------------
- name: select.int(Tensor(a) self, int dim, int index) -> Tensor(a)
  self: select_backward(grad, self.sizes(), dim, index)

@register_decomposition(aten.select_backward)
@out_wrapper()
def select_backward(grad_output: Tensor, input_sizes: List[int], dim: int, index: int):
    grad_input = grad_output.new_zeros(input_sizes)
    return torch.select_scatter(grad_input, grad_output, dim, index)

-----------------------------------------------------

- name: slice.Tensor(Tensor(a) self, int dim=0, int start=0, int end=9223372036854775807, int step=1) -> Tensor(a)
  self: slice_backward(grad, self.sizes(), dim, start, end, step)

-------------------------------------------------------
- name: index.Tensor(Tensor self, Tensor?[] indices) -> Tensor
  self: index_backward(zeros_like(self), indices, grad)
  indices: TensorList()

  Tensor index_backward(
    Tensor zeros_like_self,
    const torch::List<c10::optional<Tensor>>& indices,
    const Tensor& grad) {
  return (areAnyTensorSubclassLike({zeros_like_self, grad}) ||
          areAnyOptionalTensorSubclassLike(indices))
      ? zeros_like_self.index_put(indices, grad, true)
      : at::_index_put_impl_(zeros_like_self, indices, grad, true, true);
}

I think all of these can more or less be achieved through index and index_put
