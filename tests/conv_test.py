



#def test_shape_inference(exec_cfg: ExecutionConfig) -> None:
#
#    with TempoContext(exec_cfg, num_dims=0):
#
#        #input = torch.randn(16, 3, 32, 32, requires_grad=True)  # (B, C_in, H, W)
#        input = RecurrentTensor.rand((16, 3, 32, 32))
#
#        # Define parameters for convolution
#        stride = (2, 2)  # NOTE: changes do not work
#        padding = (1, 1)  # NOTE: changes work
#        dilation = (1, 1)  # NOTE: changes do not work
#        groups = 1
#
#        conv_layer = Conv2d(
#            in_channels=3,
#            out_channels=5,
#            kernel_size=(3, 3),
#            stride=stride,
#            padding=padding,
#            dilation=dilation,
#            groups=groups,
#            bias = False,
#            domain = (),
#        )
#
#        output : RecurrentTensor= conv_layer.forward(input) # type: ignore
#
#        expected_output_shape = (16, 5, 16, 16)
#
#        assert output.shape._shape == expected_output_shape, f"Output shape {output.shape} does not match expected shape {expected_output_shape}"


#def test_manual_gradients_orig():
#    with torch.enable_grad():
#        # Setup random seed for reproducibility
#        torch.manual_seed(0)
#
#
#        input_shape = (8, 3, 32, 32)
#        kernel_shape = (5,3,3,3)
#
#        # Define parameters for convolution
#        stride = (3, 3)  # NOTE: changes do not work
#        padding = (1, 1)  # NOTE: changes work
#        dilation = (1, 1)  # NOTE: changes do not work
#            #RuntimeError: Calculated padded input size per channel: (34 x 34).
#            #  Kernel size: (59 x 59). Kernel size can't be greater than actual input size
#        transposed = False #NOTE: True does not work (may be because we are passing in invalid input sizes)
#        output_padding = (0, 0)
#        groups = 1
#
#        # Define input and kernel
#        input = torch.randn(input_shape, requires_grad=True)  # (B, C_in, H, W)
#        kernel = torch.randn(kernel_shape, requires_grad=True)  # (C_out, C_in, KH, KW)
#
#        # Compute output using custom aten convolution
#        output_custom = torch.ops.aten.convolution(
#            input,
#            kernel,
#            None,
#            stride,
#            padding,
#            dilation,
#            transposed,
#            output_padding,
#            groups,
#        )
#
#        # Define a simple loss function
#        loss_custom = output_custom.sum()
#
#        # Compute gradients using PyTorch's autograd
#        loss_custom.backward()
#        grad_input_autograd = input.grad.clone()
#        grad_kernel_autograd = kernel.grad.clone()
#
#        # Reset gradients
#        input.grad.data.zero_()
#        kernel.grad.data.zero_()
#
#        # Manually compute gradients
#        grad_output = torch.ones_like(output_custom)
#
#        ## Gradient w.r.t. input
#        grad_input_manual = torch.ops.aten.convolution(
#            grad_output,
#            kernel,
#            None,
#            dilation,
#            padding,
#            stride,
#            not transposed,  # transposed
#            output_padding,
#            groups,
#        )
#
#
#        ## Gradient w.r.t. kernel
#        ##aten::convolution(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups)
#        grad_kernel_manual = torch.ops.aten.convolution(
#            input.permute(1, 0, 2, 3),  # Input (B, C_in, H, W) -> (C_in, B, H, W)
#            grad_output.permute(1, 0, 2, 3),  # weight (B, C_out, KH, KW) -> (C_out, B, KH, KW)
#            None,  # bias
#            stride,  # stride
#            padding,  # padding
#            dilation,  # dilation
#            transposed,  # transposed
#            output_padding,  # output_padding
#            groups,  # groups
#        ).permute(1, 0, 2, 3)
#
#
#        # Compare gradients
#        assert grad_input_autograd.shape == grad_input_manual.shape, f"Input gradients have different shapes: {grad_input_autograd.shape} != {grad_input_manual.shape}"
#        assert torch.allclose(
#            grad_input_autograd, grad_input_manual, atol=1e-5
#        ), "Input gradients do not match!"
#
#        assert grad_kernel_autograd.shape == grad_kernel_manual.shape, f"Kernel gradients have different shapes: {grad_kernel_autograd.shape} != {grad_kernel_manual.shape}"
#        assert torch.allclose(
#            grad_kernel_autograd, grad_kernel_manual, atol=1e-5
#        ), "Kernel gradients do not match!"
#
#        print("Gradients match!")
#
#
#def compute_transpose_padding(input_shape, kernel_shape, stride, padding, dilation):
#    """
#    Compute the correct padding for transposed convolution based on PyTorch's internal logic.
#    """
#    H_in, W_in = input_shape[-2:]
#    kH, kW = kernel_shape[-2:]
#    sH, sW = stride
#    pH, pW = padding
#    dH, dW = dilation
#
#    # Compute output shape using forward formula
#    H_out = (H_in + 2 * pH - dH * (kH - 1) - 1) // sH + 1
#    W_out = (W_in + 2 * pW - dW * (kW - 1) - 1) // sW + 1
#
#    # Reverse compute necessary padding
#    pH_out = (H_in - 1) * sH - H_out + kH
#    pW_out = (W_in - 1) * sW - W_out + kW
#
#    return (pH_out // 2, pW_out // 2)  # Ensure symmetric padding
#
#
#def test_manual_gradients():
#    with torch.enable_grad():
#        # Setup random seed for reproducibility
#        torch.manual_seed(0)
#
#        # Define input and kernel
#        input = torch.randn(1, 3, 32, 32, requires_grad=True)  # (B, C_in, H, W)
#        kernel = torch.randn(5, 3, 3, 3, requires_grad=True)  # (C_out, C_in, KH, KW)
#
#        # Define parameters for convolution
#        stride = (1, 1)  # NOTE: changes do not work
#        padding = (1, 1)  # NOTE: changes work
#        dilation = (1, 1)  # NOTE: changes do not work
#            #RuntimeError: Calculated padded input size per channel: (34 x 34).
#            #  Kernel size: (59 x 59). Kernel size can't be greater than actual input size
#        transposed = False #NOTE: True does not work (may be because we are passing in invalid input sizes)
#        output_padding = (0, 0)
#        groups = 1
#
#        # Compute output using custom aten convolution
#        output_custom = torch.ops.aten.convolution(
#            input,
#            kernel,
#            None,
#            stride,
#            padding,
#            dilation,
#            transposed,
#            output_padding,
#            groups,
#        )
#
#        # Define a simple loss function
#        loss_custom = output_custom.sum()
#
#        # Compute gradients using PyTorch's autograd
#        loss_custom.backward()
#        grad_input_autograd = input.grad.clone()
#        grad_kernel_autograd = kernel.grad.clone()
#
#        # Reset gradients
#        input.grad.data.zero_()
#        kernel.grad.data.zero_()
#
#        # Manually compute gradients
#        grad_output = torch.ones_like(output_custom)
#
#        # 1. Compute grad_input using col2im (equivalent to transposed convolution)
#        grad_input = torch.ops.aten.convolution(
#            grad_output,
#            kernel.flip(dims=[2,3]),  # Flip kernel spatial dimensions
#            None,
#            dilation,  # Swapped from forward stride
#            compute_transpose_padding(input.shape, kernel.shape, stride, padding, dilation),  # Corrected padding
#            stride,  # Swapped from forward dilation
#            True,  # Transposed convolution
#            (0, 0),  # output_padding (must be solved dynamically)
#            groups,
#        )
#
#        # 2. Compute grad_weight using im2col + GEMM
#        input_cols = torch.nn.functional.unfold(input, kernel.shape[-2:], dilation=dilation, padding=padding, stride=stride)
#
#        # Reshape grad_output correctly
#        grad_output_reshaped = grad_output.permute(1, 0, 2, 3).reshape(grad_output.shape[1], -1)  # (C_out, B * H_out * W_out)
#
#        # Reshape input_cols to match (B * H_out * W_out, C_in * KH * KW)
#        input_cols_reshaped = input_cols.permute(0, 2, 1).reshape(-1, input_cols.shape[1])  # (B * L, C_in * KH * KW)
#
#        # Perform GEMM to get grad_weight
#        grad_weight = grad_output_reshaped @ input_cols_reshaped  # (C_out, C_in * KH * KW)
#
#        # Reshape to match original kernel shape
#        grad_weight = grad_weight.view(kernel.shape)
#
#
#        # Compare gradients
#        assert grad_input_autograd.shape == grad_input.shape, f"Input gradients have different shapes: {grad_input_autograd.shape} != {grad_input.shape}"
#        assert torch.allclose(
#            grad_input_autograd, grad_input, atol=1e-5
#        ), "Input gradients do not match!"
#
#        assert grad_kernel_autograd.shape == grad_weight.shape, f"Kernel gradients have different shapes: {grad_kernel_autograd.shape} != {grad_weight.shape}"
#        assert torch.allclose(
#            grad_kernel_autograd, grad_weight, atol=1e-5
#        ), "Kernel gradients do not match!"
#
#        print("Gradients match!")
#
