from einops import rearrange
import torch
from torch import nn

def blockdiag_matmul (x , w ) :
    return torch . einsum (
        "bnm ,... bm - >... bn", w , x . view (* x . shape [: -1] , w . shape [0] , w . shape [ -1])
        ) . reshape (* x . shape )

class MonarchMatrix ( nn . Module ) :
    def __init__ ( self , sqrt_n : int ) :
        super () . __init__ ()
        self . sqrt_n = sqrt_n
        self . L = nn . Parameter ( torch . randn (( sqrt_n , sqrt_n , sqrt_n ) ) )
        self . R = nn . Parameter ( torch . randn (( sqrt_n , sqrt_n , sqrt_n ) ) )

    def forward ( self , x ) :
        x = rearrange (x , "... (m n) -> ... (n m)", n = self . sqrt_n )
        x = blockdiag_matmul (x , self . L )
        x = rearrange (x , "... (m n) -> ... (n m)", n = self . sqrt_n )
        x = blockdiag_matmul (x , self . R )
        return rearrange (x , " ... (m n) -> ... (n m)", n = self . sqrt_n )

class MonarchMixerLayer ( nn . Module ) :
    def __init__ ( self , sqrt_n : int , sqrt_d : int ) :
        super () . __init__ ()
        self . m1 = MonarchMatrix ( sqrt_n )
        self . m2 = MonarchMatrix ( sqrt_n )
        self . m3 = MonarchMatrix ( sqrt_d )
        self . m4 = MonarchMatrix ( sqrt_d )
        self . n_kernel = nn . Parameter ( torch . randn ( sqrt_d ** 2 , sqrt_n ** 2) )
        self . d_kernel = nn . Parameter ( torch . randn (1 , sqrt_d ** 2) )
        self . layer_norm = nn . LayerNorm ( sqrt_d ** 2)
    def forward ( self , x : torch . Tensor ) : # x. shape = (b, n, d)
        x_tilde = self . m2 ( torch . relu ( self . n_kernel * self . m1 ( x . transpose ( -1 , -2) ) )
                             ) . transpose ( -1 , -2) # mix sequence
        y = self . m4 ( torch . relu ( self . d_kernel * self . m3 ( x_tilde ) ) ) # mix features
        return self . layer_norm ( y + x_tilde ) # skip connection
