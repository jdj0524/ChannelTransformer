import torch
from einops import repeat

def hermitian(x, dim0, dim1):
    return torch.transpose(torch.conj(x), dim0, dim1)

def SumRate(u, v, h, sigma):
    # batch users dim1 dim2 complex
    
    n_users = u.shape[1]
    nr = u.shape[2]
    d = u.shape[3]
    
    u = u[:,:,:,:,0] + 1j * u[:,:,:,:,1]
    v = v[:,:,:,:,0] + 1j * v[:,:,:,:,1]
    h = h[:,:,:,:,0] + 1j * h[:,:,:,:,1]
    
    u_herm = hermitian(u, 2, 3)
    v_herm = hermitian(v, 2, 3)
    h_herm = hermitian(h, 2, 3)
    
    front = torch.matmul(u_herm, h)
    back = torch.matmul(h_herm, u)
    middle = torch.matmul(v, v_herm)
    
    
    signal_term = front @ middle @ back
    interference_term = repeat(front, "batch users dim1 dim2 -> batch users broad dim1 dim2", broad = n_users) @ repeat(middle, "batch users dim1 dim2 -> batch broad users dim1 dim2", broad = n_users) @ repeat(back, "batch users dim1 dim2 -> batch users broad dim1 dim2", broad = n_users)
    
    for i in range(n_users):
        interference_term[:, i, i, :, :] = 0
    
    interference_term = interference_term.sum(dim=2)
    noise = u_herm @ (sigma**2 * torch.eye(nr).to(interference_term.get_device()).cfloat().unsqueeze(0).unsqueeze(0)) @ u
    
    interference_term = interference_term + noise
    
    # rate_matrix = torch.eye(d).to(interference_term.get_device()) + torch.linalg.inv(interference_term) @ signal_term
    rate_matrix = torch.eye(d).to(interference_term.get_device()) + torch.linalg.solve(interference_term, signal_term, left=True)
    rate = torch.log2(torch.det(rate_matrix))
    rate = rate.sum(-1)
    return -torch.real(rate)

def SumRate_TX(u, v, h, sigma):
    # batch users dim1 dim2 complex
    
    n_users = u.shape[1]
    nr = u.shape[2]
    d = u.shape[3]
    
    
    v = v[:,:,:,:,0] + 1j * v[:,:,:,:,1]
    h = h[:,:,:,:,0] + 1j * h[:,:,:,:,1]
    
    
    v_herm = hermitian(v, 2, 3)
    h_herm = hermitian(h, 2, 3)
    
    front = h
    back = h_herm
    middle = torch.matmul(v, v_herm)
    
    
    signal_term = front @ middle @ back
    interference_term = repeat(front, "batch users dim1 dim2 -> batch users broad dim1 dim2", broad = n_users) @ repeat(middle, "batch users dim1 dim2 -> batch broad users dim1 dim2", broad = n_users) @ repeat(back, "batch users dim1 dim2 -> batch users broad dim1 dim2", broad = n_users)
    
    for i in range(n_users):
        interference_term[:, i, i, :, :] = 0
    
    interference_term = interference_term.sum(dim=2) + sigma**2 * torch.eye(nr).to(interference_term.get_device())
    
    # interference_term = sigma**2 * torch.eye(nr).to(interference_term.get_device())
    
    # rate_matrix = torch.eye(nr).to(interference_term.get_device()) + torch.linalg.inv(interference_term) @ signal_term
    rate_matrix = torch.eye(nr).to(interference_term.get_device()) + torch.linalg.solve(interference_term, signal_term, left=False)
    rate = torch.log2(torch.det(rate_matrix))
    rate = rate.sum(-1).real
    return rate

def Interference(u, v, h, sigma):
    
    n_users = u.shape[1]
    
    u = u[:,:,:,:,0] + 1j * u[:,:,:,:,1]
    v = v[:,:,:,:,0] + 1j * v[:,:,:,:,1]
    h = h[:,:,:,:,0] + 1j * h[:,:,:,:,1]
    
    interference_term = repeat(hermitian(u, -1, -2), "batch users dim1 dim2 -> batch users broad dim1 dim2", broad = n_users) @ repeat(h, "batch users dim1 dim2 -> batch users broad dim1 dim2", broad = n_users) @ repeat(v, "batch users dim1 dim2 -> batch broad users dim1 dim2", broad = n_users)
    
    for i in range(n_users):
        interference_term[:, i, i, :, :] = 0
        
    interference_term = torch.norm(interference_term.sum(dim = 2), dim = (-1,-2))
    return interference_term

def Interference_TX(u, v, h, sigma):
    
    n_users = u.shape[1]
    
    u = u[:,:,:,:,0] + 1j * u[:,:,:,:,1]
    v = v[:,:,:,:,0] + 1j * v[:,:,:,:,1]
    h = h[:,:,:,:,0] + 1j * h[:,:,:,:,1]
    
    interference_term = repeat(h, "batch users dim1 dim2 -> batch users broad dim1 dim2", broad = n_users) @ repeat(v, "batch users dim1 dim2 -> batch broad users dim1 dim2", broad = n_users)
    
    for i in range(n_users):
        interference_term[:, i, i, :, :] = 0
        
    interference_term = torch.norm(interference_term.sum(dim = 2), dim = (-1,-2))
    return interference_term