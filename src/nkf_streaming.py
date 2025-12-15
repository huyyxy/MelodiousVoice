
import torch
import torch.nn as nn
import numpy as np

class NKFStreaming(nn.Module):
    """
    NKF model wrapper for streaming (real-time) inference.
    Maintains internal states for STFT and RNNs.
    """
    def __init__(self, model, block_size=1024, hop_size=256):
        super().__init__()
        self.model = model
        self.L = model.L
        self.block_size = block_size
        self.hop_size = hop_size
        
        # Internal buffers for OLA (Overlap-Add)
        self.in_buffer_x = None
        self.in_buffer_y = None
        self.out_buffer = None
        
        # Filter states
        self.h_prior = None
        self.h_posterior = None
        self.X_history = None
        
        # STFT window
        self.register_buffer('window', torch.hann_window(block_size))
        
        # RNN Hidden States
        self.h_rr = None
        self.h_ir = None
        self.h_ri = None
        self.h_ii = None
    
    def reset_state(self, device):
        # Reset buffers and states
        self.in_buffer_x = torch.zeros(self.block_size - self.hop_size, device=device)
        self.in_buffer_y = torch.zeros(self.block_size - self.hop_size, device=device)
        self.out_buffer = torch.zeros(self.block_size, device=device)
        
        self.h_prior = None
        self.h_posterior = None
        self.X_history = None
        
        # Initialize RNN states
        # B=1, F=513 (for 1024 FFT)
        F = self.block_size // 2 + 1
        rnn_layers = self.model.kg_net.rnn_layers
        rnn_dim = self.model.kg_net.rnn_dim
        
        self.h_rr = torch.zeros(rnn_layers, F, rnn_dim, device=device)
        self.h_ir = torch.zeros(rnn_layers, F, rnn_dim, device=device)
        self.h_ri = torch.zeros(rnn_layers, F, rnn_dim, device=device)
        self.h_ii = torch.zeros(rnn_layers, F, rnn_dim, device=device)

    def process_chunk(self, x_chunk, y_chunk):
        """
        Process a chunk of time-domain audio.
        x_chunk: Reference signal chunk (Tensor) - shape (hop_size,)
        y_chunk: Microphone signal chunk (Tensor) - shape (hop_size,)
        Returns: Echo-cancelled output chunk (Tensor) - shape (hop_size,)
        """
        device = x_chunk.device
        
        if self.in_buffer_x is None:
            self.reset_state(device)

        # 1. Update Input Buffers (Shift and Append)
        # We process one frame at a time. 
        # The buffer holds (block_size - hop_size) previous samples.
        # We append the new hop_size samples to form a full block_size window.
        curr_x_window = torch.cat([self.in_buffer_x, x_chunk])
        curr_y_window = torch.cat([self.in_buffer_y, y_chunk])
        
        # Update buffers for next iteration
        self.in_buffer_x = curr_x_window[self.hop_size:]
        self.in_buffer_y = curr_y_window[self.hop_size:]

        # 2. STFT
        # Manual STFT for one frame
        X_frame = torch.fft.rfft(curr_x_window * self.window)
        Y_frame = torch.fft.rfft(curr_y_window * self.window)
        
        # Dimensions: (F,) where F = block_size/2 + 1
        # Add batch and time dimensions: (B=1, F, T=1) for compatibility
        X_stft = X_frame.unsqueeze(0).unsqueeze(2) # (1, F, 1)
        Y_stft = Y_frame.unsqueeze(0).unsqueeze(2)
        
        # 3. NKF Logic
        B, F, T = X_stft.shape 
        # Note here B=1, but effectively we are processing F independent frequency bins in parallel?
        # Actually in NKF implementation:
        # B is usually batch size. F is frequency.
        # The code flattens B*F. 
        # In our case, we have 1 "audio batch", so B=1.
        # So we work with shape (F, ...)
        
        if self.h_prior is None:
             self.h_prior = torch.zeros(B * F, self.L, 1, dtype=torch.complex64, device=device) # (F, L, 1)
             self.h_posterior = torch.zeros(B * F, self.L, 1, dtype=torch.complex64, device=device)
             self.X_history = torch.zeros(B * F, self.L, 1, dtype=torch.complex64, device=device)

        # Shift X history
        # current_x_flat: (F, 1, 1)
        current_x_flat = X_stft.view(B * F, 1, 1)
        self.X_history = torch.cat([current_x_flat, self.X_history[:, :-1, :]], dim=1)
        xt = self.X_history # (F, L, 1)
        
        # Skip if silence (energy check)
        if xt.abs().mean() >= 1e-5:
            dh = self.h_posterior - self.h_prior
            self.h_prior = self.h_posterior
            
            curr_y_flat = Y_stft.view(B * F) # (F,)
            
            # e = y - x^H * h
            # xt.transpose(1,2) -> (F, 1, L)
            # h_prior -> (F, L, 1)
            # matmul -> (F, 1, 1) -> squeeze -> (F,)
            e = curr_y_flat - torch.matmul(xt.transpose(1, 2), self.h_prior).squeeze()
            
            # input_feature for KGNet
            # cat [ xt (F,L,1) -> (F,L), e (F,) -> (F,1), dh (F,L,1) -> (F,L) ]
            # final dim=1: L + 1 + L = 2L + 1
            input_feature = torch.cat([xt.squeeze(-1), e.unsqueeze(1), dh.squeeze(-1)], dim=1)
            
            # KGNet pass
            # We need to manually call KGNet components to inject hidden state
            # self.model.kg_net(input_feature) uses its own init_hidden which resets state.
            # So we replicate kg_net logic here using our persisted states.
            
            kg_net = self.model.kg_net
            feat = kg_net.fc_in(input_feature).unsqueeze(1) # (F, 1, fc_dim)
            
            # Helper to run GRU step with state
            # complex_gru(input, h_rr, h_ir, h_ri, h_ii)
            # input: (F, 1, input_size)
            # h_rr: (layers, F, hidden_size)
            rnn_out, self.h_rr, self.h_ir, self.h_ri, self.h_ii = kg_net.complex_gru(
                feat, self.h_rr, self.h_ir, self.h_ri, self.h_ii
            )
            
            kg = kg_net.fc_out(rnn_out).permute(0, 2, 1) # (F, L, 1)
            
            # Update posterior
            self.h_posterior = self.h_prior + torch.matmul(kg, e.unsqueeze(-1).unsqueeze(-1))

        # Compute Output Echo Estimate
        echo_hat = torch.matmul(xt.transpose(1, 2), self.h_posterior).squeeze() # (F,)
        
        # 4. ISTFT & Overlap-Add
        S_hat_frame = Y_stft.view(F) - echo_hat # (F,)
        
        # IRFFT
        s_frame = torch.fft.irfft(S_hat_frame) * self.window
        
        # Add to output buffer
        self.out_buffer += s_frame
        
        # Extract the valid output chunk
        output_chunk = self.out_buffer[:self.hop_size].clone()
        
        # Shift output buffer
        self.out_buffer = torch.roll(self.out_buffer, -self.hop_size)
        self.out_buffer[-self.hop_size:] = 0
        
        return output_chunk
