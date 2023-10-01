import torch
import numbers
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import litdata
import torchvision.transforms.functional as tvF

from scipy.sparse import coo_matrix as cpu_coo_matrix
from scipy.sparse import issparse
from scipy.sparse.csgraph import connected_components as cpu_concom

from typing import Union, Tuple, Iterable, Optional, Callable, Sequence


### General helper functions

def check_kernel():
    '''Function that checks if the venv kernel for JupyterLab is correctly installed.
    
    This function essentially instructs you on the status of the availability of the custom
    virtual environment kernel in your JupyterLab instance. The stdout/print output either
    confirms that the kernel is correctly installed, or intructs you on how to configure the 
    venv/kernel for in5310.
    '''
    from ipykernel.kernelspec import KernelSpecManager
    if not 'in5310' in KernelSpecManager().find_kernel_specs():
        print(
            'Error: IN5310 environment not found!',
            '\nRun the install script by executing the command:', 
            '\n\n!/projects/ec232/venvs/install_in5310_kernel.sh',
            '\n\nin a cell of your notebook.',
            '\nAfter that, refresh your browser window and select the "in5310" kernel',
            'in the top right corner.\nThis setup should only be required once per user.',
        )
    else:
        print(
            'IN5310 environment already installed.',
            'Select the "in5310" kernel in the top right corner of your notebook.',
        )


def unravel_index(indices:torch.Tensor, shape:Union[Iterable[int], torch.Tensor]) -> torch.Tensor:
    '''Converts a tensor of flat indices into a tensor of coordinate vectors.

    Args:
        index (torch.Tensor): Indices to unravel.
        shape (tuple[int]): Shape of tensor.

    Returns:
        torch.Tensor: Tensor (long) of unraveled indices.
    '''
    try:
        shape = indices.new_tensor(torch.Size(shape))[:,None] # type: ignore
    except Exception:
        pass
    shape = F.pad(shape, (0,0,0,1), value=1)                  # type: ignore
    coefs = shape[1:].flipud().cumprod(dim=0).flipud()
    return torch.div(indices[None], coefs, rounding_mode='trunc') % shape[:-1]


def sample_inverse(n_samples:int, cdf:torch.Tensor, support:torch.Tensor) -> torch.Tensor:
    '''Uses inverse sampling trick to sample from a flattened cdf.
    
    NOTE: The support can be 2D, but ordered by the flattened cdf.
    
    Args:
        n_samples (int): Number of samples to draw.
        cdf (torch.Tensor): Monotonically increasing cumulative density function.
        support (torch.Tensor): Support of cdf, first dim. must match len(cdf)
        
    Returns:
        Samples from support given cdf.
    '''
    assert cdf.ndim == 1
    assert support.shape[0] == cdf.shape[0]
    rand = torch.rand(n_samples)
    return support[torch.searchsorted(cdf, rand)]


### General Classes

class CrossEntropyLoss(nn.Module):
    """Implements Cross Entropy Loss with optional label smoothing.

    The label smoothing can be applied to prevent the model from
    being over-confident in its predictions. It provides more
    information to the model, which may lead to better generalization.

    Attributes:
        use_smoothing (bool): Whether to apply label smoothing.
        eps (float): Smoothing factor applied to each class.
        negative (float): Value assigned to non-target classes in smoothing.
        positive (float): Value assigned to the target class in smoothing.
    """
    
    def __init__(self, smoothing:float, classes:int):
        """Initializes the CrossEntropyLoss class.

        Args:
            smoothing (float): Smoothing factor, must be in [0, 1].
            classes (int): The number of classes in the target distribution.
        """
        super().__init__()
        self.use_smoothing = smoothing > 0.
        self.eps = smoothing / classes
        self.negative = self.eps
        self.positive = (1 - smoothing) + self.eps

    def smoothing_loss(self, pred:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        """Calculates the smoothed cross-entropy loss.

        Args:
            pred (torch.Tensor): Predictions tensor of shape (batch_size, num_classes).
            target (torch.Tensor): Ground truth tensor. Can be either one-hot encoded or
                class indices of shape (batch_size,).

        Returns:
            torch.Tensor: The smoothed cross-entropy loss.
        """
        if target.ndim == 1:
            true_dist = torch.full_like(pred, self.negative)
            true_dist.scatter_(1, target.data.unsqueeze(1), self.positive)

        else:
            assert pred.shape == target.shape, 'pred.shape != target.shape!'
            true_dist = target
            zeromask = target <= 0
            true_dist[zeromask].fill_(self.negative)
            true_dist[~zeromask].mul_(self.positive)
        
        return torch.sum(-true_dist * pred.log_softmax(dim=1), dim=1).mean()

    def standard_loss(self, pred:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        """Calculates the standard cross-entropy loss.

        Args:
            pred (torch.Tensor): Predictions tensor of shape (batch_size, num_classes).
            target (torch.Tensor): Ground truth tensor. Can be either one-hot encoded or
                class indices of shape (batch_size,).

        Returns:
            torch.Tensor: The standard cross-entropy loss.
        """
        if target.ndim == 1:
            return F.nll_loss(pred.log_softmax(dim=1), target)
        else:
            assert pred.shape == target.shape, 'pred.shape != target.shape!'
            return torch.sum(-target * pred.log_softmax(dim=1), dim=1).mean()

    def forward(self, pred, target):
        """Computes the loss based on whether smoothing is enabled.

        Calls either the smoothing_loss method or the standard_loss method.

        Args:
            pred (torch.Tensor): Predictions tensor of shape (batch_size, num_classes).
            target (torch.Tensor): Ground truth tensor. Can be either one-hot encoded or
                class indices of shape (batch_size,).

        Returns:
            torch.Tensor: The computed loss, either standard or smoothed.
        """
        if self.use_smoothing:
            return self.smoothing_loss(pred, target)
        return self.standard_loss(pred, target)


#### W1: ViTs

class UnNormalize(object):
    """Un-normalizes tensor after torchvision.transforms.Normalize(mean, std) has been applied."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):

        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)

        return tensor


#### W2: GNNs

# TESTING

_GNN_TEST_STDOUT = True


def test_simplegcn(module):
    sgcn = module(24, 24, activation=nn.Identity())
    sgcn.linear.weight.data = torch.load('./source/gcn_lweights.pt')
    sgcn.linear.bias.data = torch.load('./source/gcn_lbias.pt')
    vertices = torch.load('./source/gcn_vertices.pt')
    edges = torch.load('./source/gcn_edges.pt')
    edge_attr = torch.load('./source/gcn_edge_attr.pt')
    sampledata = torch.load('./source/gcn_lout.pt')
    with torch.no_grad():
        result = sgcn(vertices, edges, edge_attr)
        msg = 'Test failed! Check your implementation!\n'
        msg += f'{result}\n'
        msg += '-' * 80 + '\n'
        msg += f'{sampledata}'
        if not torch.allclose(result, sampledata, atol=1e-6):
            raise ValueError(msg)
    if _GNN_TEST_STDOUT:
        print('Successful test! Your SimpleGCN module is working as expected :)')
    return


def test_gcn(module):
    global _GNN_TEST_STDOUT
    _GNN_TEST_STDOUT = False
    try:
        test_simplegcn(module)
    except e:
        _GNN_TEST_STDOUT = True
        raise e
    _GNN_TEST_STDOUT = True
    print('Successful test! Your GCN module is working as expected :)')
    return


def test_gin(module):
    gin = module(24, 24, activation=nn.Identity())
    gin.gcn.linear.weight.data = torch.load('./source/gcn_lweights.pt')
    gin.gcn.linear.bias.data = torch.load('./source/gcn_lbias.pt')
    gin.linear.weight.data = torch.load('./source/gcn_lweights.pt')
    gin.linear.bias.data = torch.load('./source/gcn_lbias.pt')
    vertices = torch.load('./source/gcn_vertices.pt')
    edges = torch.load('./source/gcn_edges.pt')
    edge_attr = torch.load('./source/gcn_edge_attr.pt')
    sampledata = torch.load('./source/gin_lout.pt')
    with torch.no_grad():
        result = gin(vertices, edges, edge_attr)
        msg = 'Test failed! Check your implementation!\n'
        msg += f'{result}\n'
        msg += '-' * 80 + '\n'
        msg += f'{sampledata}'
        if not torch.allclose(result, sampledata, atol=1e-6):
            raise ValueError(msg)
    print('Successful test! Your GIN module is working as expected :)')
    return



# Helper functions

def default_sparse_vals(A_ij:torch.Tensor, A_val:Optional[torch.Tensor]=None) -> torch.Tensor:
    '''Function for adding default values to sparse matrix.

    Args:
        A_ij (torch.Tensor): The indices of non-zero elements of the sparse matrix.
        A_val (Optional[torch.Tensor]): The values of the non-zero elements in the sparse matrix.

    Returns:
        torch.Tensor: Values of non-zero elements. If none are provided, returns ones.
    '''
    if A_val is None:
        A_val = torch.ones_like(A_ij[0], dtype=torch.float)
    return A_val


def get_degree_matrix(order:int, A_ij:torch.Tensor) -> torch.Tensor:
    '''Computes the degree matrix for an adjacency matrix.
    
    NOTE: This uses the directed scheme for universality; in other words, 
          we count both in- and outdegrees for the graph, even if it is
          undirected.
    
    Args:
        order (int): The total number of vertices in the graph.
        A_ij (torch.Tensor): The indices of non-zero elements of the adjacency matrix.
        
    Returns:
        torch.Tensor: The degree matrix.
    '''
    i, j = A_ij
    ones_m = default_sparse_vals(A_ij, None)
    degree = A_ij.new_zeros(order, dtype=torch.float)
    degree.scatter_add_(0, i, ones_m)
    degree.scatter_add_(0, j, ones_m)
    return degree


def add_self_loops(
    order:int, A_ij:torch.Tensor, A_val:torch.Tensor, 
    fill_value:Union[float, torch.Tensor]=1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''Adds self loops to the graph.

    Args:
        order (int): The total number of vertices in the graph.
        A_ij (torch.Tensor): The indices of non-zero elements of the adjacency matrix.
        A_val (Optional[torch.Tensor]): The values of the non-zero elements in the adjacency matrix.
        fill_value (Union[float,torch.Tensor]): Optional value to fill self loops with. Defaults to 1.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The updated sparse adjacency, including self loops.
    '''
    i, j = A_ij
    non_loop_mask = i != j
    A_ij = A_ij[:, non_loop_mask]
    A_val = A_val[non_loop_mask]
    I_n = torch.arange(order, device=A_ij.device).unsqueeze(0).expand(2, -1)
    val_n = A_val.new_ones(order) * fill_value
    A_ij = torch.cat([A_ij, I_n], -1)
    A_val = torch.cat([A_val, val_n], -1)
    return A_ij, A_val


def scatter_softmax(
    values:torch.Tensor, index:torch.Tensor, num_rows:int
) -> torch.Tensor:
    '''Computes a scattered softmax function.

    Args:
        values (torch.Tensor): Values to compute over.
        index (torch.Tenosr): Indices to compute over.
        num_rows (int): Number of rows, used to initialize outputs.

    Returns:
        torch.Tensor: Output of softmax.
    '''
    with torch.no_grad():
        amax = values.new_zeros(num_rows, values.shape[-1])
        amax.scatter_reduce_(0, index, values, 'amax')
    values = (values - amax.gather(0, index)).exp()
    den = values.new_zeros(num_rows, values.shape[-1])
    den.scatter_reduce_(0, index, values, 'sum')
    return values / den.gather(0, index).add(1e-7)


class GCN(nn.Module):
    
    '''A Graph Convolutional Network (GCN) module using PyTorch sparse tensors.

    Attributes:
        in_dim (int): The dimensionality of the input features.
        out_dim (int): The dimensionality of the output features.
        act (Callable): The activation function to use after the linear transformation.
        linear (nn.Linear): A linear transformation layer.
        loop_fill (float): The value to fill self-loops with, i.e., 1.0.
    '''
    
    def __init__(self, in_dim:int, out_dim:int, activation:Callable=nn.ReLU()):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.act = activation
        self.linear = nn.Linear(in_dim, out_dim)
        self.loop_fill = 1.0
        
    def get_invsqrt_D(self, order:int, A_ij:torch.Tensor) -> torch.Tensor:
        '''Computes the self adjoint inverse square root of the degree matrix.

        Args:
            order (int): The total number of vertices in the graph.
            A_ij (torch.Tensor): The indices of non-zero elements of the adjacency matrix.

        Returns:
            torch.Tensor: The inverse square root of the degree matrix.
        '''        
        i, j = A_ij
        degree = get_degree_matrix(order, A_ij)
        return 1 / (degree[i] * degree[j]).sqrt()
        
    def compute(
        self, vertices:torch.Tensor, edges:torch.Tensor, 
        edge_attr:torch.Tensor
    ) -> torch.Tensor:
        '''Performs the graph convolution operation. 
        
        This function uses PyTorch sparse tensors for improved efficiency.

        Args:
            vertices (torch.Tensor): The features of the vertices.
            edges (torch.Tensor): The edges of the graph.
            edge_attr (torch.Tensor): The attributes of the edges.

        Returns:
            torch.Tensor: The updated vertex features after applying the GCN.
        '''
        order = vertices.shape[0]
        A = torch.sparse_coo_tensor(edges, edge_attr, size=(order, order))
        output = A @ self.linear(vertices)
        return output
    
    def forward(
        self, vertices:torch.Tensor, edges:torch.Tensor, 
        edge_attr:Optional[torch.Tensor]=None
    ) -> torch.Tensor:
        '''Runs a forward pass through the GCN.

        Args:
            vertices (torch.Tensor): The features of the vertices.
            edges (torch.Tensor): The edges of the graph.
            edge_attr (Optional[torch.Tensor]): The attributes of the edges.

        Returns:
            torch.Tensor: The updated vertex features after applying the GCN and the activation function.
        '''
        order, in_dim = vertices.shape
        assert in_dim == self.in_dim
        assert (edges.min() >= 0) and (edges.max() <= order)

        edge_attr = default_sparse_vals(edges, edge_attr)        
        edges, edge_attr = add_self_loops(order, edges, edge_attr, self.loop_fill)        
        invsqrt_D = self.get_invsqrt_D(order, edges)
        edge_attr.mul_(invsqrt_D)
        vertices = self.compute(vertices, edges, edge_attr)
        return self.act(vertices)

    
class GraphWoofIterator:
    
    '''Class for working with the GraphWoof dataset.
    '''
    
    def __init__(
        self, batch_size:int=64, device:torch.device=torch.device('cpu'),
        litdatapath:str='/projects/ec232/data/'
    ):
        import torchvision.transforms as T
        self.batch_size = batch_size
        self.device = device
        
        # Define the modalities we want to retrieve 
        override_extensions = ('V.npy', 'E.npy', 'cls')
        vis_override_extensions = ('V.npy', 'E.npy', 'cls', 'jpg', 'seg16')

        # Define how we process the modalities for training and validation
        postprocess = [torch.tensor, torch.tensor, nn.Identity()]
        vis_postprocess = [
            torch.tensor, self.add_global_edges, torch.tensor, 
            T.ToTensor(), lambda x: torch.tensor(x.astype(np.int64))
        ]
        
        # Create training, validation, and visualization dataset
        self.traindata = litdata.LITDataset(
            'GraphWoof', litdatapath, 
            override_extensions=override_extensions
        ).map_tuple(*postprocess)
        
        self.valdata = litdata.LITDataset(
            'GraphWoof', litdatapath, train=False, 
            override_extensions=override_extensions
        ).map_tuple(*postprocess)
        
        self.visdata = litdata.LITDataset(
            'GraphWoof', litdatapath, train=False, 
            override_extensions=vis_override_extensions
        ).map_tuple(*vis_postprocess)
        
        # Create loaders
        self.trainloader = torch.utils.data.DataLoader(
            self.traindata, batch_size=batch_size, sampler=None,
            num_workers=0, pin_memory=True, pin_memory_device=device,
            collate_fn=self.graph_collate,
            drop_last=True
        )
        self.valloader = torch.utils.data.DataLoader(
            self.valdata, batch_size=batch_size, sampler=None,
            num_workers=0, pin_memory=True, pin_memory_device=device,
            collate_fn=self.graph_collate,
            drop_last=True
        )
              
    def training(self):
        '''Constructs an iterator for training.
        
        Returns:
            tuple: (index, (vertex_features, edges, global_index, labels))
        '''
        with self.traindata.shufflecontext():
            for i, batch in enumerate(self.trainloader):
                v, e, gi, l = [s.to(self.device) for s in batch]
                yield i, (v, e, gi, l)
            
    def validation(self):
        '''Constructs an iterator for validation.
        
        NOTE: This does not turn off gradient calculations, which should be
              done manually under the context `with torch.no_grad():`.
              This choice is mainly for pedagogical reasons.
        
        Returns:
            tuple: (index, (vertex_features, edges, global_index, labels))
        '''
        for i, batch in enumerate(self.valloader):
            v, e, gi, l = [s.to(self.device) for s in batch]
            yield i, (v, e, gi, l)
        
        
    def get_random_visualization_sample(self):
        '''Retrieves a random visualization example from the dataset.
        
        This can be used to retrieve a sample for verification and visualization.
        
        Returns:
            tuple: (vertex_features, edges, global_index, labels, image, segmentation)
        '''
        with self.visdata.shufflecontext():
            v, e, l, im, seg = [s.to(self.device) for s in self.visdata[0]]
        gi = l.new_zeros(1)
        return (v, e, gi, l, im, seg)
        
    
    @staticmethod
    def add_global_edges(E:Sequence):
        '''Adds global class tokens to the graph edges.

        By default, the edges in GraphWoof only include the edges between
        the regions. When using a global readout token, we connect this to all the
        nodes in the graph.

        Args:
            E (torch.Tensor): Default edges.

        Returns:
            torch.Tensor: Edges including global tokens.
        '''
        if not torch.is_tensor(E): # If edges are numpy array.
            E = torch.tensor(E)
        maxval = E.max().item()
        nodes = torch.arange(1, maxval, device=E.device)
        cls_edges = torch.stack([torch.zeros_like(nodes), nodes])
        return torch.cat([E, cls_edges], -1)
    
    def graph_collate(self, batch):
        '''Performs graph collation for a batch of samples.

        This function performs the graph collation. Since graphs have a variable
        number of vertices and edges, we join all graphs in a batch as one single 
        graph with disconnected images and process them in parallel.

        Args:
            batch (tuple): Tuple with vertex features, edges, and labels.

        Returns:
            (tuple[torch.Tensor...]): Collated batch with graph features.
        '''
        add_edges = GraphWoofIterator.add_global_edges
        features = torch.cat([g[0] for g in batch], 0)
        labels = torch.tensor([g[-1] for g in batch])
        edges = []
        global_index = []
        current_index = 0
        for graph in batch:
            global_index.append(current_index)
            edges.append(add_edges(graph[1]) + current_index)
            current_index += graph[0].shape[0]
        return features.arcsinh(), torch.cat(edges, -1), torch.tensor(global_index), labels


#### W4: VAEs

def test_reparameterize_gaussian(reparameterize_gaussian:Callable):
    correct_z = torch.load('./source/vae_sample_z.pt')
    mean = torch.load('./source/vae_sample_mean.pt')
    logvar = torch.load('./source/vae_sample_logvar.pt')
    torch.manual_seed(0)
    z = reparameterize_gaussian(mean, logvar)
    assert torch.equal(z, correct_z), f"Your implementation does not match the teacher's solution. Expected {correct_z}, but got {z}"
    print('Successful test! Your reparameterize_gaussian function is working as expected :)')  

def test_vae_loss(vae_loss:Callable):
    x = torch.load('./source/vae_sample_x.pt')
    x_hat = torch.load('./source/vae_sample_x_hat.pt')
    mean = torch.load('./source/vae_sample_mean.pt')
    logvar = torch.load('./source/vae_sample_logvar.pt')
    correct_loss = torch.load('./source/vae_loss.pt')
    loss = vae_loss(x, x_hat, mean, logvar)
    assert torch.equal(loss, correct_loss), f"Your implementation does not match the teacher's solution. Expected {correct_loss}, but got {loss}"
    print('Successful test! Your vae_loss function is working as expected :)')

def test_vae(module):
    VAE = module(z_dim=2, n_channels=1)
    x = torch.load('./source/vae_sample_x.pt')
    z = VAE.get_latents(x)
    assert z.shape == torch.Size([1, 2]), f"Your implementation does not match the teacher's solution. Expected the encoder output to have shape {torch.Size([1, 2])} but got {z.shape}"
    print('Successful test! Your Encoder and Decoder implementation is working as expected :)')
    
#### W5: NFs 

class ToRGBTensor:
    
    def __call__(self, img):
        return tvF.to_tensor(img).expand(3, -1, -1) # Expand to 3 channels
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def sample_week5(n_samples:int):
    cdf = torch.load('./source/nf_cdf.pt')
    support = unravel_index(torch.arange(512**2), (512,512)).mT / 512
    samples = sample_inverse(n_samples, cdf, support).flip(1)
    samples[:,1].sub_(1).mul_(-1)
    return samples

class DatasetMoons:
    def sample(self, n, seed=None):
        moons = make_moons(n_samples=n, noise=0.05, random_state=seed)[0].astype(np.float32)
        return torch.from_numpy(moons)

# Code from sklearn source (to avoid installing sklearn in venv)
def make_moons(n_samples=100, *, shuffle=True, noise=None, random_state=None):
    if isinstance(n_samples, numbers.Integral):
        n_samples_out = n_samples // 2
        n_samples_in = n_samples - n_samples_out
    else:
        try:
            n_samples_out, n_samples_in = n_samples
        except ValueError as e:
            raise ValueError(
                "`n_samples` can be either an int or a two-element tuple."
            ) from e

    generator = check_random_state(random_state)

    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5

    X = np.vstack(
        [np.append(outer_circ_x, inner_circ_x), np.append(outer_circ_y, inner_circ_y)]
    ).T
    y = np.hstack(
        [np.zeros(n_samples_out, dtype=np.intp), np.ones(n_samples_in, dtype=np.intp)]
    )

    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

    if noise is not None:
        X += generator.normal(scale=noise, size=X.shape)

    return X, y

def resample(*arrays, replace=True, n_samples=None, random_state=None, stratify=None):
    """Resample arrays or sparse matrices in a consistent way."""
    max_n_samples = n_samples
    random_state = check_random_state(random_state)

    if len(arrays) == 0:
        return None

    first = arrays[0]
    n_samples = first.shape[0] if hasattr(first, "shape") else len(first)

    if max_n_samples is None:
        max_n_samples = n_samples
    elif (max_n_samples > n_samples) and (not replace):
        raise ValueError(
            "Cannot sample %d out of arrays with dim %d when replace is False"
            % (max_n_samples, n_samples)
        )

    check_consistent_length(*arrays)

    if stratify is None:
        if replace:
            indices = random_state.randint(0, n_samples, size=(max_n_samples,))
        else:
            indices = np.arange(n_samples)
            random_state.shuffle(indices)
            indices = indices[:max_n_samples]
    else:
        # Code adapted from StratifiedShuffleSplit()
        y = check_array(stratify, ensure_2d=False, dtype=None)
        if y.ndim == 2:
            # for multi-label y, map each distinct row to a string repr
            # using join because str(row) uses an ellipsis if len(row) > 1000
            y = np.array([" ".join(row.astype("str")) for row in y])

        classes, y_indices = np.unique(y, return_inverse=True)
        n_classes = classes.shape[0]

        class_counts = np.bincount(y_indices)

        # Find the sorted list of instances for each class:
        # (np.unique above performs a sort, so code is O(n logn) already)
        class_indices = np.split(
            np.argsort(y_indices, kind="mergesort"), np.cumsum(class_counts)[:-1]
        )

        n_i = _approximate_mode(class_counts, max_n_samples, random_state)

        indices = []

        for i in range(n_classes):
            indices_i = random_state.choice(class_indices[i], n_i[i], replace=replace)
            indices.extend(indices_i)

        indices = random_state.permutation(indices)

    # convert sparse matrices to CSR for row-based indexing
    arrays = [a.tocsr() if issparse(a) else a for a in arrays]
    resampled_arrays = [_safe_indexing(a, indices) for a in arrays]
    if len(resampled_arrays) == 1:
        # syntactic sugar for the unit argument case
        return resampled_arrays[0]
    else:
        return resampled_arrays

def util_shuffle(*arrays, random_state=None, n_samples=None):
    """Shuffle arrays or sparse matrices in a consistent way."""
    return resample(
        *arrays, replace=False, n_samples=n_samples, random_state=random_state
    )

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    :class:`numpy:numpy.random.RandomState`
        The random state object based on `seed` parameter.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )
    
def check_consistent_length(*arrays):
    """Check that all arrays have consistent first dimensions.
    Checks whether all objects in arrays have the same shape or length.
    """

    lengths = [_num_samples(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: %r"
            % [int(l) for l in lengths]
        )
        
def _num_samples(x):
    """Return number of samples in array-like x."""
    message = "Expected sequence or array-like, got %s" % type(x)
    if hasattr(x, "fit") and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError(message)

    if not hasattr(x, "__len__") and not hasattr(x, "shape"):
        if hasattr(x, "__array__"):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, "shape") and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError(
                "Singleton array %r cannot be considered a valid collection." % x
            )
        # Check that shape is returning an integer or default to len
        # Dask dataframes may not return numeric shape[0] value
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]

    try:
        return len(x)
    except TypeError as type_error:
        raise TypeError(message) from type_error
        
def _safe_indexing(X, indices, *, axis=0):
    """Return rows, items or columns of X using indices.
    """
    if indices is None:
        return X

    if axis not in (0, 1):
        raise ValueError(
            "'axis' should be either 0 (to index rows) or 1 (to index "
            " column). Got {} instead.".format(axis)
        )

    indices_dtype = _determine_key_type(indices)

    if axis == 0 and indices_dtype == "str":
        raise ValueError("String indexing is not supported with 'axis=0'")

    if axis == 1 and X.ndim != 2:
        raise ValueError(
            "'X' should be a 2D NumPy array, 2D sparse matrix or pandas "
            "dataframe when indexing the columns (i.e. 'axis=1'). "
            "Got {} instead with {} dimension(s).".format(type(X), X.ndim)
        )

    if axis == 1 and indices_dtype == "str" and not hasattr(X, "loc"):
        raise ValueError(
            "Specifying the columns using strings is only supported for "
            "pandas DataFrames"
        )

    if hasattr(X, "iloc"):
        return _pandas_indexing(X, indices, indices_dtype, axis=axis)
    elif hasattr(X, "shape"):
        return _array_indexing(X, indices, indices_dtype, axis=axis)
    else:
        return _list_indexing(X, indices, indices_dtype)
    
def _determine_key_type(key, accept_slice=True):
    """Determine the data type of key."
    """
    err_msg = (
        "No valid specification of the columns. Only a scalar, list or "
        "slice of all integers or all strings, or boolean mask is "
        "allowed"
    )

    dtype_to_str = {int: "int", str: "str", bool: "bool", np.bool_: "bool"}
    array_dtype_to_str = {
        "i": "int",
        "u": "int",
        "b": "bool",
        "O": "str",
        "U": "str",
        "S": "str",
    }

    if key is None:
        return None
    if isinstance(key, tuple(dtype_to_str.keys())):
        try:
            return dtype_to_str[type(key)]
        except KeyError:
            raise ValueError(err_msg)
    if isinstance(key, slice):
        if not accept_slice:
            raise TypeError(
                "Only array-like or scalar are supported. A Python slice was given."
            )
        if key.start is None and key.stop is None:
            return None
        key_start_type = _determine_key_type(key.start)
        key_stop_type = _determine_key_type(key.stop)
        if key_start_type is not None and key_stop_type is not None:
            if key_start_type != key_stop_type:
                raise ValueError(err_msg)
        if key_start_type is not None:
            return key_start_type
        return key_stop_type
    if isinstance(key, (list, tuple)):
        unique_key = set(key)
        key_type = {_determine_key_type(elt) for elt in unique_key}
        if not key_type:
            return None
        if len(key_type) != 1:
            raise ValueError(err_msg)
        return key_type.pop()
    if hasattr(key, "dtype"):
        try:
            return array_dtype_to_str[key.dtype.kind]
        except KeyError:
            raise ValueError(err_msg)
    raise ValueError(err_msg)
    
def _array_indexing(array, key, key_dtype, axis):
    """Index an array or scipy.sparse consistently across NumPy version."""
    if issparse(array) and key_dtype == "bool":
        key = np.asarray(key)
    if isinstance(key, tuple):
        key = list(key)
    return array[key] if axis == 0 else array[:, key]



def show_two_images(img1, img2, figsize=(12,6), titles=('',''), cmap=None):
    fig, axs = plt.subplots(1,2, figsize=figsize)
    axs[0].imshow(img1, cmap=cmap)
    axs[0].axis('off')
    axs[0].set_title(titles[0])
    axs[1].imshow(img2, cmap=cmap)
    axs[1].axis('off')
    axs[1].set_title(titles[1])
    return fig, axs

def test_AffineFlow(module):
    flow = module(dim=2, flip=True)
    x = torch.load('source/w5_x_moon.pt')
    flow.s_func = torch.load('source/w5_flow_s_func.pt')
    flow.t_func = torch.load('source/w5_flow_t_func.pt')
    flow.scale = torch.load('source/w5_flow_scale.pt')

    z, logdet = flow(x)
    assert z.shape == torch.Size([100, 2]), f"The output z of AffineFlow.forward(x) should have shape torch.Size([100, 2]), but is {z.shape}"
    assert logdet.shape == torch.Size([100]), f"The output logdet of AffineFlow.forward(x) should have shape torch.Size([100]), but is {logdet.shape}"

    z_corr = torch.load('source/w5_z_moon.pt')
    logdet_corr = torch.load('source/w5_logdet_moon.pt')
    assert torch.allclose(z, z_corr), f"Your z does not match the teacher's implementation.\nz={z}\ncorrect={z_corr}"
    assert torch.allclose(logdet, logdet_corr), f"Your logdet does not match the teacher's implementation."


def test_RealNVP(module):
    x = torch.load('source/w5_x_moon.pt')
    nf = module(dim=2, num_flows=1, prior_loc=torch.zeros(2), prior_cov=torch.eye(2))
    assert len(nf.flows) == 1
    nf.flows[0].s_func = torch.load('source/w5_flow_s_func.pt')
    nf.flows[0].t_func = torch.load('source/w5_flow_t_func.pt')
    nf.flows[0].scale = torch.load('source/w5_flow_scale.pt')
    nf.flows[0].flip = True

    z_list, logdet, prior_logprob = nf(x)
    assert len(z_list) == 2, f"Expected len(z_list)=2, but got len(z_list){len(z_list)}"
    assert z_list[-1].shape == torch.Size([100, 2]), f"The output z of RealNVP.forward(x) should have shape torch.Size([100, 2]), but is {z.shape}"
    assert logdet.shape == torch.Size([100]), f"The output logdet of RealNVP.forward(x) should have shape torch.Size([100]), but is {logdet.shape}"
    assert prior_logprob.shape == torch.Size([100]), f"The output prior_logprob of RealNVP.forward(x) should have shape torch.Size([100]), but is {logdet.shape}"

    z_corr = torch.load('source/w5_z_moon.pt')
    logdet_corr = torch.load('source/w5_logdet_moon.pt')
    assert torch.allclose(z_list[-1], z_corr), f"Your z does not match the teacher's implementation."
    assert torch.allclose(logdet, logdet_corr), f"Your logdet does not match the teacher's implementation."
    

class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes*stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet18Enc(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, n_channels=3):
        super().__init__()
        self.in_planes = 64
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)
        self.linear = nn.Linear(512, z_dim)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        z = self.linear(x)
        return z

class ResNet18Dec(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, img_size=[64, 64],  n_channels=3):
        """
        img_size: [n_channels, height, width] or [height, width]. If given as length 3, the n_channels variable will be ignored.
        n_channels: Number of channels in the input image.
        """
        super().__init__()
        if len(img_size) == 3:
            self.n_channels = img_size[0]
            self.img_size = img_size[1:]
        else: 
            self.n_channels = n_channels
            self.img_size = img_size
        
        self.interpolation_scale = int(img_size[0]/2**4)
        
        self.in_planes = 512

        self.linear = nn.Linear(z_dim, 512)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, n_channels, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=self.interpolation_scale)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.sigmoid(self.conv1(x))
        x = F.interpolate(x, size=self.img_size, mode='bilinear') # Resize if needed?
        x = x.view(x.size(0), self.n_channels, self.img_size[0], self.img_size[1])
        return x

class ResAE(nn.Module):
    """ Auto-Encoder with residual encoder and decoder """

    def __init__(self, z_dim, n_channels=3, img_size=[64,64]):
        """
        img_size: [n_channels, height, width] or [height, width]. If given as length 3, the n_channels variable will be ignored.
        n_channels: Number of channels in the input image.
        """
        super().__init__()
        
        self.latent_dim = z_dim
        if len(img_size) == 3:
            self.n_channels = img_size[0]
            self.img_size = img_size[1:]
        else: 
            self.n_channels = n_channels
            self.img_size = img_size

        self.encoder = ResNet18Enc(z_dim=self.latent_dim, n_channels=self.n_channels)
        self.decoder = ResNet18Dec(z_dim=self.latent_dim, img_size=self.img_size, n_channels=self.n_channels)

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x, z

    def get_latents(self, x):
        """ Encode data x into latent representations z. """
        z = self.encoder(x)
        return z

#### W6: SBGMs

def sample_week6(n_samples:int):
    cdf = torch.load('./source/sbgm_cdf.pt')
    support = unravel_index(torch.arange(512**2), (512,512)).mT / 512
    samples = sample_inverse(n_samples, cdf, support).flip(1)
    samples[:,1].sub_(1).mul_(-1)
    return samples

#### W10: XAI

def adjust_saturation(rgb:torch.Tensor, mul:float):
    '''Adjusts saturation via interpolation / extrapolation.

    Args:
        rgb (torch.Tensor): An input tensor of shape (..., 3) representing the RGB values of an image.
        mul (float): Saturation adjustment factor. A value of 1.0 will keep the saturation unchanged.

    Returns:
        torch.Tensor: A tensor of the same shape as the input, with adjusted saturation.
    '''    
    weights = rgb.new_tensor([0.299, 0.587, 0.114])
    grayscale = (
        torch.matmul(rgb, weights)
            .unsqueeze(dim=-1)
            .expand_as(rgb)
            .to(dtype=rgb.dtype)
    )
    return torch.lerp(grayscale, rgb, mul).clip(0,1)


def peronamalik1(img, niter=5, kappa=0.0275, gamma=0.275):
    '''Anisotropic diffusion.
    
    Perona-Malik anisotropic diffusion type 1, which favours high contrast 
    edges over low contrast ones.
    
    `kappa` controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence diffusion
    across step edges.  A large value reduces the influence of intensity
    gradients on conduction.
           
    Reference:
    P. Perona and J. Malik.
    Scale-space and edge detection using ansotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence,
    12(7):629-639, July 1990.

    Args:
        img (torch.Tensor): input image
        niter (int): number of iterations
        kappa (float): conduction coefficient.
        gamma (float): controls speed of diffusion (generally max 0.25)
    
    Returns:
    Diffused image.    
    '''
    
    deltaS, deltaE = img.new_zeros(2, *img.shape)
    
    for _ in range(niter):
        deltaS[...,:-1,:] = torch.diff(img, dim=-2)
        deltaE[...,:,:-1] = torch.diff(img, dim=-1)

        gS = torch.exp(-(deltaS/kappa)**2.)
        gE = torch.exp(-(deltaE/kappa)**2.)
        
        S, E = gS*deltaS, gE*deltaE

        S[...,1:,:] = S.diff(dim=-2)
        E[...,:,1:] = E.diff(dim=-1)
        img = img + gamma*(S+E)
    
    return img


def rgb_to_ycbcr(feat: torch.Tensor, dim=-1) -> torch.Tensor:
    '''Convert RGB features to YCbCr.

    Args:
        feat (torch.Tensor): Pixels to be converted YCbCr.

    Returns:
        torch.Tensor: YCbCr converted features.
    '''    
    r,g,b = feat.unbind(dim)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    delta = 0.5
    cb = (b - y) * 0.564 + delta
    cr = (r - y) * 0.713 + delta
    return torch.stack([y, cb, cr], dim)


def fast_uidx_1d(ar:torch.Tensor) -> torch.Tensor:
    '''Pretty fast unique index calculation for 1d tensors.

    Args:
        ar (torch.Tensor): Tensor to compute unique indices for.

    Returns:
        torch.Tensor: Tensor (long) of indices.
    '''
    assert ar.ndim == 1, f'Need dim of 1, got: {ar.ndim}!'
    perm = ar.argsort()
    aux = ar[perm]
    mask = ar.new_zeros(aux.shape[0], dtype=torch.bool)
    mask[:1] = True
    mask[1:] = aux[1:] != aux[:-1]
    return perm[mask]


def fast_uidx_long2d(ar:torch.Tensor) -> torch.Tensor:
    '''Pretty fast unique index calculation for 2d long tensors (row wise).

    Args:
        ar (torch.Tensor): Tensor to compute unique indices for.

    Returns:
        torch.Tensor: Tensor (long) of indices.
    '''
    assert ar.ndim == 2, f'Need dim of 2, got: {ar.ndim}!'
    m = ar.max() + 1
    r, c = ar
    cons = r*m + c
    return fast_uidx_1d(cons)


def scatter_add_1d(src:torch.Tensor, idx:torch.Tensor, n:int) -> torch.Tensor:
    '''Computes scatter add with 1d source and 1d index.

    Args:
        src (torch.Tensor): Source tensor.
        idx (torch.Tensor): Index tensor.
        n (int): No. outputs.

    Returns:
        torch.Tensor: Output tensor.
    '''
    assert src.ndim == 1
    assert len(src) == len(idx)
    out = src.new_zeros(n)
    return out.scatter_add_(0, idx, src)


def scatter_mean_2d(src:torch.Tensor, idx:torch.Tensor) -> torch.Tensor:
    '''Computes scatter mean with 2d source and 1d index over first dimension.

    Args:
        src (torch.Tensor): Source tensor.
        idx (torch.Tensor): Index tensor.

    Returns:
        torch.Tensor: Output tensor.
    '''
    assert src.ndim == 2
    assert len(src) == len(idx)
    if idx.ndim == 1:
        idx = idx.unsqueeze(1).expand(*src.shape)
    out = src.new_zeros(idx.max()+1, src.shape[1]) # type: ignore
    return out.scatter_reduce_(0, idx, src, 'mean', include_self=False)

    
def cosine_similarity_argmax(
    vfeat:torch.Tensor, edges:torch.Tensor, sizes:torch.Tensor, 
    nnz: int, lvl: Optional[int]=None, valbit:int=16
) -> torch.Tensor:    
    '''Compute the cosine similarity between edge-connected vertex features 
    
    Uses Bit-Packing to perform argmax. We pack the first `valbit` bits in a
    signed `int64` (torch.long) tensor, and the indices are packed in the
    remaining bits (defaults to 47). Note that this implementation is device 
    agnostic, but could result in some computational overhead from storing all
    similarities.
    
    NOTE: Original code from Superpixel Transformers by Aasan et al. 2023.
    
    Args:
        vfeat (torch.Tensor): Vertex features of shape (N, D), where N is the
            number of vertices and D is the dimension of the feature vector.
        edges (torch.Tensor): Edge indices as a tuple of tensors (u, v), where
            u and v are 1-D tensors containing source and target vertex
            indices, respectively.
        sizes (torch.Tensor): Sizes tensor used to normalize similarity measures.
        nnz (int): Number of non-zero values.
        lvl (Optional[int]): Level parameter to adjust the mean normalization,
            defaults to None, in which case the mean is computed from sizes.
        valbit (int): Number of bits used to represent the packed similarity
            value, defaults to 16.

    Returns:
        torch.Tensor: Packed similarity values as a tensor of long integers.
            The packed format includes both similarity values and vertex
            indices, and is suitable for further processing or storage.

    Note:
        The function includes assertions to ensure that the resulting packed
        values are within valid bounds. The compressed format allows for efficient
        storage and manipulation of large graph structures with associated
        similarity measures.

    Examples:
        >>> vfeat = torch.rand((100, 50))
        >>> edges = (torch.randint(100, (200,)), torch.randint(100, (200,)))
        >>> sizes = torch.rand(100)
        >>> nnz = 200
        >>> packed_sim = cosine_similarity_argmax(vfeat, edges, sizes, nnz)
    '''
    idxbit = 63 - valbit
    u, v = edges
    sfl = sizes.contiguous().to(dtype=vfeat.dtype)

    if lvl is None:
        mu = sfl.mean()
    else:
        mu = sfl.new_tensor(4**(lvl-1))

    std = sfl.std().clip(min=1e-6)
    
    stdwt = ((sfl - mu) / std)
    sim = torch.where(
        u == v, 
        stdwt.clip(-.75, .75)[u], 
        torch.cosine_similarity(vfeat[u], vfeat[v], -1, 1e-4)
    ).clip(-1, 1)
    
    shorts = (((sim + 1.0) / 2) * (2**valbit - 1)).long()
    packed_u = (shorts << idxbit) | u
    packed_v = (shorts << idxbit) | v
    packed_values = torch.zeros(nnz, dtype=torch.long, device=v.device)
    packed_values.scatter_reduce_(0, v, packed_u, 'amax', include_self=False)
    packed_values.scatter_reduce_(0, u, packed_v, 'amax', include_self=True)  
    out = packed_values & (2**(idxbit)-1)
    
    assert (out.max().item() < nnz)
    assert (out.min().item() >= 0)
    return out


def get_superpixel_segmentation(img : torch.Tensor) -> torch.Tensor:
    '''Custom graph based superpixel segmentation.
    
    Hierarchically builds a superpixel segmentation in 5 levels. In this
    example, only the top level is extracted.
    
    There are lots of optimized code packed into one big function here, for
    more details, email `mariuaas(at)ifi.uio.no.
    
    NOTE: Original code from Superpixel Transformers by Aasan et al. 2023.
    
    Args:
        img (torch.Tensor): Input image.
    
    Returns:
        torch.Tensor: Superpixel segmentation.
    '''
    device = img.device
    dtype = img.dtype

    shape_proc = lambda x: x.permute(1,0,2,3).reshape(x.shape[1], -1).unbind(0)
    center_ = lambda x: x.mul_(2).sub_(1)

    def contrast1_(x, mu, lambda_):
        '''Kuwaraswamy contrast.
        '''
        x.clip_(0,1)
        m, a = x.new_tensor(mu).clip_(0, 1), x.new_tensor(lambda_).clip_(0)
        b = -(x.new_tensor(2)).log_() / (1-m**a).log_()
        return x.pow_(a).mul_(-1).add_(1).pow_(b).mul_(-1).add_(1)  

    def contrast2_(x, lambda_):
        '''Arcsinh contrast.
        '''
        if lambda_ == 0:
            return x
        tmul = x.new_tensor(lambda_)
        m, d = tmul, torch.arcsinh(tmul)
        if lambda_ > 0:
            return x.mul_(m).arcsinh_().div_(d)
        return x.mul_(d).sinh_().div_(m)
    
    def dgrad(img, lambda_):
        '''Discrete gradients with Scharr Kernel.
        '''
        img = img.mean(1, keepdim=True)
        kernel = img.new_tensor([[[[-3.,-10,-3.],[0.,0.,0.],[3.,10,3.]]]])
        kernel = torch.cat([kernel, kernel.mT], dim=0)
        out = F.conv2d(
            F.pad(img, 4*[1], mode='replicate'), 
            kernel, 
            stride=1
        ).div_(16)
        return contrast2_(out, lambda_)
    
    def col_transform(colfeat, shape, lambda_col):
        '''Color normalization.
        '''
        device = colfeat.device
        b, _ , h, w = shape
        c = colfeat.shape[-1]
        f = adjust_saturation(colfeat.add(1).div_(2), 2.718)
        f = rgb_to_ycbcr(f, -1).mul_(2).sub_(1)
        contrast2_(f, lambda_col)
        return peronamalik1(
            f.view(b, h, w, c).permute(0,3,1,2).cpu(),
            4, 
            0.1,
            0.5
        ).permute(0,2,3,1).view(-1, c).clip_(-1,1).to(device)
        
    def spstep(
        lab:torch.Tensor, edges:torch.Tensor, vfeat:torch.Tensor, 
        sizes:torch.Tensor, nnz:int, lvl:int,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, 
        int, torch.Tensor
    ]:
        '''Superpixel edge contraction step.
        '''
        # Compute argmax over cosine similarities
        sim = cosine_similarity_argmax(vfeat, edges, sizes, nnz, lvl=lvl)
        
        # Connected Components (performed on CPU)
        ones = torch.ones_like(lab, device='cpu').numpy()
        adj = (lab.cpu().numpy(), sim.cpu().numpy())
        csr = cpu_coo_matrix((ones, adj), shape=(nnz,nnz)).tocsr()
        cc = lab.new_tensor(cpu_concom(csr)[1]).to(device)

        # Update Parameters
        vfeat_new = scatter_mean_2d(vfeat, cc)
        edges_new = cc[edges].contiguous()
        edges_new = edges_new[:, fast_uidx_long2d(edges_new)]
        lab_new = cc.unique()
        nnz_new = len(lab_new)
        sizes_new = scatter_add_1d(sizes, cc, nnz_new)
        return lab_new, edges_new, vfeat_new, sizes_new, nnz_new, cc
    
    # Intialize Parameters
    lvl = 0
    maxlvl = 5
    lambda_grad = 27.8
    lambda_col = 10.
    batch_size, _, height, width = img.shape
    nnz = batch_size * height * width
    
    # Initialize Segmentation
    labels = torch.arange(nnz, device=device)
    lr = labels.view(batch_size, height, width).unfold(-1, 2, 1).reshape(-1, 2).mT
    ud = labels.view(batch_size, height, width).unfold(-2, 2, 1).reshape(-1, 2).mT
    edges = torch.cat([lr, ud], -1)
    sizes = torch.ones_like(labels)
    hierograph = [labels]

    # Preprocess Features
    den = max(height, width)
    r, g, b = shape_proc(img.clone())
    center_(contrast1_(r, .485, .539))
    center_(contrast1_(g, .456, .507))
    center_(contrast1_(b, .406, .404))
    gy, gx = shape_proc(dgrad(img, lambda_grad))
    features = torch.stack([r,g,b,gy,gx], -1)
    maxgrad = contrast2_(img.new_tensor(13/16), lambda_grad).mul_(2**.5)
    features = torch.cat([
        col_transform(features[:,:3], img.shape, lambda_col), 
        center_(features[:,-2:].norm(2, dim=1, keepdim=True).div_(maxgrad)),
    ], -1).float()
            
    # Construct superpixel hierarchy    
    while lvl < maxlvl:
        lvl += 1
        labels, edges, features, sizes, nnz, cc = spstep(
            labels, edges, features, sizes, nnz, lvl
        )
        hierograph.append(cc)
    
    # Collapse hierarchy to top level
    segmentation = hierograph[0]
    for i in range(1, lvl + 1):
        segmentation = hierograph[i][segmentation]
        
    return segmentation.view(batch_size, height, width)
