# LT-PINN: Lagrangian topology-conscious physics-informed neural network for boundary-focused engineering optimization

Yuanye Zhou b a, Zhaokun Wang ${}^{b, c}$ , Kai Zhou b, Avi Tang b, Xiaofan Li b ${}^{c, e}$

${}^{a}$ Department of Civil and Environmental Engineering, The Hong Kong Polytechnic University, Hong Kong, China

${}^{\mathrm{b}}$ Department of Mechanical Engineering, The Hong Kong Polytechnic University, Hong Kong, China

${}^{\mathrm{c}}$ School of Fashion and Textiles, The Hong Kong Polytechnic University, Hong Kong, China

${}^{\mathrm{d}}$ Research Institute for Sustainable Urban Development (RISUD), The Hong Kong Polytechnic University, Hong Kong, China

e Department of Mechanical Engineering, The University of Hong Kong, Hong Kong, China

## A R T I C L E I N F O

Keywords:

PINN

PDEs

Lagrangian topology optimization

Meshless

## A B S T R A C T

Physics-informed neural networks (PINNs) have emerged as a powerful meshless tool for topology optimization, capable of simultaneously determining optimal topologies and physical solutions. However, conventional PINNs rely on density-based topology descriptions, which necessitate manual interpolation and limit their applicability on precise topology boundary and its normal reconstruction. To address this, we propose Lagrangian topology-conscious PINNs (LT-PINNs), a novel framework for boundary-focused engineering optimization. By parameterizing the control variables of topology boundary curves as learnable parameters, LT-PINNs eliminate the need for manual interpolation and enable precise boundary determination. We further introduce specialized boundary condition loss function and topology loss function to ensure sharp and accurate boundary representations, even for intricate topologies. The accuracy and robustness of LT-PINNs are validated via two types of partial differential equations (PDEs), including elastic equation with Dirichlet boundary conditions and Laplace's equation with Neumann boundary conditions. To demonstrate its broad applicability, we also implemented LT-PINNs on several primitive topologies and benchmarked its performance. The effectiveness of LT-PINNs is finally verified on more complex time-dependent and time-independent flow problems without relying on measurement data, and showcase their engineering application potential in flow velocity rearrangement, transforming a uniform upstream velocity into a sine-shaped downstream profile. The results demonstrate (1) LT-PINNs achieve substantial reductions in relative ${L}_{2}$ errors compared with the state-of-art density topology-oriented PINNs (DT-PINNs), (2) LT-PINNs can handle arbitrary boundary conditions, making them suitable for a wide range of PDEs, and (3) LT-PINNs can infer clear topology boundaries without manual interpolation, especially for complex topologies.

## 1. Introduction

Topology optimization is a design optimization technique that determines the ideal material layout, within a specified design space, to meet desired performance criteria, while satisfying constraints related to boundary conditions [1]. It has been widely applied to many industry applications [2], such as aerospace [3], automotive [4], energy [5], and bioengineering [6]. While, topology optimization is a computationally intensive process, as it usually requires hundreds of design iterations. In each iteration, the physical field must be solved to obtain the sensitivity information needed to update the current design. Large-scale topology optimization problems are extremely computationally intensive, as they may involve millions or more design variables [7]. To manage the substantial computational demands, several classical techniques are developed in the past decades, such as parallel computing [8,9], advanced iterative solvers [10], multi-scale or multi-resolution methods [11,12].

---

* Corresponding author.

E-mail address: cee-kai.zhou@polyu.edu.hk (K. Zhou).

---

Recently, machine learning has emerged as a rapidly developing paradigm for topology optimization, offering faster and more efficient solutions compared to classical techniques [13-17]. However, purely data-driven machine learning models lack the ability to encode physical information, which can lead to overfitting. In contrast, physics-informed neural networks (PINNs) are especially attractive for topology optimization because they integrate physical information directly into the model. This integration allows for simultaneous determination of optimal topology and solution of physical fields, eliminating the need for iterative processes required by classical techniques and mitigating the overfitting problem in data-driven models. Another notable advantage of PINNs in topology optimization is that they are meshless methods, which are capable of capturing fine geometric details with significant changes in shapes and topologies during the optimization process. So far, PINNs have been successfully applied to solve a wide range of partial differential equations (PDEs) problems [18], such as heat problems [19], high-speed compressible flow [20], vortex-induced vibration [21] and linear elastic problems [22].

In a prior study of topology optimization, Chen et al. [23] utilized PINN to address the inverse design of photonic metamaterials. They innovatively set both the permittivity map and the PDE solution as outputs of the PINN, enabling the network to simultaneously reconstruct the permittivity distribution and predict the PDE solution. This approach effectively integrated the physical constraints of the unknown metamaterial topology, i.e. permittivity map, into the neural network framework. In a subsequent study, Lu et al. [24] proposed a hard-constrained PINN which exactly imposes Dirichlet and periodic boundary conditions into the PINN to improve model accuracy and convergence speed. For structural topology optimization, Jeong et al. [25] developed a density topology-orient PINN that integrates density field into neural network and was able to handle topology optimization without measurement data. However, the aforementioned PINNs for topology optimization predominantly depend on density-based descriptions, such as permittivity maps, density fields, and solid fractions. These descriptions, being inherently continuous outputs of PINNs, only provide an implicit representation of the actual topology. As a result, manual interpolation is required to identify the topology boundary, which complicates human understanding and hinders the engineering reproduction process. Additionally, current PINNs are restricted to Dirichlet boundary conditions because they are unable to accurately determine the boundary location until the optimization process is finalized.

To address the limitation on implicit topology representations, several techniques have been proposed to enhance the capabilities of PINNs based on sharpened density-based descriptions. Jeong et al. [25] applied a high-order power scaling factor on density field to boost its convergence towards either 0 or 1 , so as to identify topology boundary. Furthermore, Jeong et al. [26] proposed Fourier mapping and periodic activation function to enhance the convergence and accuracy of PINNs using implicit representations. Zhu et al. [27] used a sigmoid activation function on the density field to minimize the buffer region between fluid and solid topology. Mowlavi & Kamrin [28] suggested using Eikonal regularization to enforce a uniform density field everywhere in order to ensure consistency of physical laws across the topology boundary. Yin et al. [17] utilized Gaussian interpolation to obtain a more accurate material property from the density field. It is worth mentioning that generating accurate topology boundaries is key to ensuring manufacturability for many applications, such as the optimization of simple pipe-shaped geometries [29], thin-wall-shaped foldable structures [30], and coating structures [31,32]. While these techniques indeed enhanced the contrast of boundaries, they require sophisticated adjustments to alter the inherently continuous nature of density-based descriptions. In some cases, this limitation can impede the accurate identification of topological boundaries and their normals, often necessitating additional manual parameter tuning.

In contrast, to provide a clear topology boundary, some classical topology optimization techniques employ a different approach, known as the Lagrangian topology description [33]. The Lagrangian topology description utilizes a set of moving morphable patches to construct the topology. It is based on the concept that a series of overlapping, simple, and parameterizable patches can be used to represent an object with highly complex topology. This approach provides an explicit method for topology optimization and has led to significant advancements in structural design [34-38]. For instance, Guo et al. [39] used B-spline curve to describe the topology boundary of the void and optimize void distribution and shapes. However, classical Lagrangian topology optimization usually requires special meshing schemes, such as adaptive mesh scheme [40], during topology updates, as conventional mesh regeneration for new topologies is computationally expensive.

As meshless methods, PINNs show strong potential for integration with Lagrangian topology descriptions. This synergy could significantly enhance topology optimization efficiency while preserving well-defined boundaries. However, despite this promise, no prior work has successfully merged PINNs with the Lagrangian approach, leaving both the validity and computational advantages of such a hybrid model unexplored. Moreover, achieving seamless integration between these two frameworks presents substantial theoretical and implementation challenges. To bridge this gap and advance PINN-based topology optimization, the primary objective of this study is to propose Lagrangian Topology-conscious PINNs (LT-PINNs), a novel framework that incorporates the control parameters of topology boundary curves as learnable variables. Our approach introduces (1) boundary condition loss function for both Dirichlet and Neumann conditions, enabling precise boundary identification and PDE solution; and (2) topology loss function to handle geometrically complex structures. This algorithmic innovation delivers a new class of PINNs specifically designed for high-fidelity topology optimization, particularly in scenarios requiring explicit boundary representation. These unique features, which differentiate the proposed approach from state-of-the-art methods, represent its main novelty and contribution.

The remaining of the paper is organized as follows: The methodological principles underlying our proposed approach and its benchmark counterpart are thoroughly examined with theoretical details in Section 2. A series of numerical experiments spanning diverse physical problem domains are presented and compared in Section 3 to highlight the strength of the proposed approach, followed by the conclusion given by Section 4.

## 2. Methodologies

This section begins with a review of underlying principles for PINNs, followed by the introduction of a typical class of PINNs for topology optimization, which will serve as the benchmark method in this study. Adopting the architecture from Zhu et al. [27], we term this benchmark method DT-PINNs (density topology-oriented PINNs) to highlight its focus on density topology optimization. The proposed Lagrangian topology-conscious PINNs (LT-PINNs) with key advanced features are subsequently introduced.

### 2.1. DT-PINNs

For any general partial differential equation (PDE) expressed as:

$$
{\mathcal{N}}_{\omega }\left\lbrack  {\mathbf{u}\left( {\mathbf{x}, t}\right) }\right\rbrack   = f\left( {\mathbf{x}, t}\right) ,\mathbf{x} \in  {R}^{D}, \tag{1}
$$

where $\mathcal{N}$ denotes any differential operator, $x$ and $t$ are the spatial and temporal coordinates, respectively, $D$ is the dimension of space, $u$ is the solution, $\omega$ is a parameter that defines the differential operator, and $f$ is a known function.

In forward PDE problems, the parameter $\omega$ is known, and the goal is to solve Eq. (1) subject to the given initial and boundary conditions. Conversely, in inverse problems, $\omega$ is unknown. The objective is thus to determine the solution to Eq. (1) and to infer the value of $\omega$ based on available data for $u$ and the equation. Specifically, for topology optimization, the unknown $\omega$ refers to the topology representation in terms of density function $\rho$ . Ideally, it is a Dirac delta function as below:

$$
\rho \left( {\mathbf{x}, t}\right)  = \left\{  \begin{array}{ll} 0, & \mathbf{x} \in  {\Omega }_{0} \\  1, & \mathbf{x} \in  {\Omega }_{1}, \end{array}\right. \tag{2}
$$

where ${\Omega }_{1}$ denotes the topology occupied domain, and ${\Omega }_{0}$ denotes the rest of the domain.

Level-set methods are widely adopted in topology optimization [41]. Integrating the level-set function into PINNs thus remains a promising strategy. Notably, the density function in Eq. (2) is derived from the level-set function ( $\phi$ ) [42], which defines the signed distance from a point $x$ to the boundary $\Gamma$ , such that $\rho  = 0$ for $\phi  < 0$ and $\rho  = 1$ for $\phi  > 0$ . An ideal density function is a discontinuous Dirac delta function, while a level-set function is inherently continuous. However, a neural network approximation of the density function is inherently smooth and continuous. As a result, the learned density function can be interpreted as a generalized level-set function that decays rapidly from 1 to 0 across the boundary.

Since 0 and 1 can be flipped, depending on the focus of partial differential equation, Eq. (1) is then modified as:

$$
\left( {1 - \rho \left( {\mathbf{x}, t}\right) }\right) \left( {{\mathcal{N}}_{{\omega }^{ * }}\left\lbrack  {\mathbf{u}\left( {\mathbf{x}, t}\right) }\right\rbrack   - f\left( {\mathbf{x}, t}\right) }\right)  = 0, \tag{3}
$$

where ${\omega }^{ * }$ represents the known components of $\omega$ , such as fluid viscosity, Young’s modulus, Poisson’s ratio, and so on.

The objective of topology optimization is to determine the full solution $u$ to Eq. (3) and infer $\rho$ subject to sparse measurement data of $u$ and PDE, as shown in Fig. 1(a). Before optimization, within the region of interest (ROI), the topology is unknown and only sparse measurement data is known. After topology optimization, both the topology and full solution data of PDE can be determined.

In order to achieve the objective, DT-PINNs are adopted from Zhu et al. [27], as illustrated in Fig. 1(b). In DT-PINNs, there exists a feed-forward neural network (FNN) which takes $x$ and $t$ as inputs to approximate the solution to $u$ and $\rho$ . With the automatic differentiation, the PDE can then be encoded to the neural networks. The total loss function for training the DT-PINNs is expressed as follows:

$$
{L}_{tot}\left( \mathbf{\theta }\right)  = {\lambda }_{p}{L}_{p} + {\lambda }_{d}{L}_{d}
$$

$$
{L}_{p} = \frac{1}{N}\mathop{\sum }\limits_{{i = 1}}^{N}{R}_{i}^{2},{R}_{i} = {\left( 1 - \rho \left( \mathbf{x}, t\right) \right) }_{i}{\left( {\mathcal{N}}_{{\omega }^{ * }}\left\lbrack  \mathbf{u}\left( \mathbf{x}, t\right) \right\rbrack   - f\left( \mathbf{x}, t\right) \right) }_{i}, \tag{4}
$$

$$
{L}_{d} = \frac{1}{{N}_{d}}\mathop{\sum }\limits_{{i = 1}}^{{N}_{d}}{\left( \mathbf{u} - {\mathbf{u}}^{ * }\right) }_{i}^{2},
$$

where ${L}_{\text{ tot }}$ is the total loss, ${L}_{p}$ is the PDE loss, ${L}_{d}$ is the data loss, $\theta$ represents the parameters in the neural networks, $R$ is the residual of PDEs, ${\mathbf{u}}^{ * }$ is the sparse measurements/initial conditions on $\mathbf{u}, N$ is the number of sample points for PDE loss, and ${N}_{d}$ is the number of sample points for data loss. ${\lambda }_{p}$ and ${\lambda }_{d}$ are the weights of the PDE loss and data loss respectively.

DT-PINNs encodes physics-information of PDEs via PDE loss $\left( {L}_{p}\right)$ in its total loss function $\left( {L}_{tot}\right)$ . When ${L}_{tot}$ is zero, it implies that both $u$ and $\rho$ are well solved, and topology (represented by $\rho  = 1$ , as defined in Eq. (2)) can be inferred. In general, the total loss is minimized during training through stochastic gradient descent optimization.

Note that in DT-PINNs, the inferred $\rho$ must remain a continuous scalar function to approximate the Dirac delta function, as required by automatic differentiation. Following training completion, a threshold value $\left( {\rho }^{ * }\right)$ between 0 and 1 needs to be manually selected to identify the topology boundary. Due to the sharp transition in $\rho$ across this boundary, the resulting topology exhibits significant sensitivity to the chosen threshold value. Furthermore, DT-PINNs struggle to enforce boundary conditions since the topology boundary only becomes discernible after training. While this limitation can be partially mitigated through composite PDE loss function incorporating Dirichlet boundary conditions, as proposed in Zhu et al. [27], the fundamental challenge remains. The Dirichlet boundary conditions incorporated PDE loss function is mathematically described as:

$$
{L}_{p} = \frac{1}{N}\mathop{\sum }\limits_{{i = 1}}^{N}{R}_{i}^{2}
$$

$$
{R}_{i} = {\left( 1 - \widehat{\rho }\left( \mathbf{x}, t\right) \right) }_{i}{\left( {\mathcal{N}}_{{\omega }^{ * }}\left\lbrack  \mathbf{u}\left( \mathbf{x}, t\right) \right\rbrack   - f\left( \mathbf{x}, t\right) \right) }_{i} + \tag{5}
$$

$$
\widehat{\rho }{\left( \mathbf{x}, t\right) }_{i}{\left( \mathbf{u}\left( \mathbf{x}, t\right)  - {\mathbf{u}}_{\mathbf{b}}^{ * }\left( \mathbf{x}, t\right) \right) }_{i},
$$

$$
\widehat{\rho } = \frac{1}{1 + {e}^{-{c\rho }}},
$$

![3_269_156_1168_883_0.jpg](images/3_269_156_1168_883_0.jpg)

Fig. 1. Illustration of topology optimization via DT-PINNs: a) the objective of topology optimization; b) the framework of DT-PINNs.

where $\widehat{\rho }$ is normalized $\rho , c$ is a constant parameter, and ${\mathbf{u}}_{b}^{ * }$ is the known Dirichlet boundary conditions. In this study, we use Eq. (5) as the PDE loss function for DT-PINNs with $c = {10}$ , following the suggestion from Zhu et al. [27]. Nevertheless, such composite PDE loss function cannot accommodate boundary conditions requiring known boundary normal directions, such as Neumann boundary conditions.

Overall, DT-PINNs are positioned within the broader PINN family by their flexibility with complex geometries. Nevertheless, their limitation lies in the diffuse boundaries lacking sharp definitions and normals, which often requires manual intervention and hinders manufacturability of the design.

### 2.2. LT-PINNs

To overcome the key limitations of DT-PINNs, specifically their reliance on manual boundary determination and limited adaptability to boundary conditions, we propose LT-PINNs built upon Lagrangian topology optimization approach to significantly improve the manufacturability of generated topologies. Its framework is illustrated in Fig. 2. This new approach allows one to determine the full solution data of PDEs and infer topology subject to sparse measurement data, PDEs, and prior random distributed topology patches. The fundamental idea relies on constructing complex topologies through compositions of simple, overlapping, and parameterized patches. Since each patch admits an explicit mathematical representation, the topology boundary can be determined in each training step, allowing boundary conditions to be rigorously enforced through corresponding loss functions.

![4_269_154_1171_986_0.jpg](images/4_269_154_1171_986_0.jpg)

Fig. 2. Illustration of topology optimization via LT-PINNs: a) the objective of Lagrangian topology optimization; b) the framework of LT-PINNs.

In LT-PINNs to map the learnable boundary curve back to the density function used in solving PDEs, a geometric distance function (i.e., $\delta \left( {x,\gamma }\right)$ ) is introduced to reformulate Eq. (3) as:

$$
\left( {\mathop{\prod }\limits_{{k = 1}}^{n}{\delta }^{k}\left( {\mathbf{x},\mathbf{\gamma }}\right) }\right) \left( {{\mathcal{N}}_{{\omega }^{ * }}\left\lbrack  {\mathbf{u}\left( {\mathbf{x}, t}\right) }\right\rbrack   - f\left( {\mathbf{x}, t}\right) }\right)  = 0, \tag{6}
$$

Here, $\delta \left( {\mathbf{x},\mathbf{\gamma }}\right)$ is defined as:

$$
\delta \left( {\mathbf{x},\mathbf{\gamma }}\right)  = \frac{1}{1 + {e}^{-\beta  \cdot  {SDF}}},
$$

$$
{SDF} = \operatorname{sign}\left( \mathbf{x}\right)  \cdot  d\left( {\mathbf{x},{F}_{\gamma }}\right) ,
$$

$$
\operatorname{sign}\left( x\right)  = \left\{  \begin{array}{ll}  - 1, & \text{ if }x \in  {\Omega }_{1}, \\  0, & \text{ if }x \in  \partial {\Omega }_{1}, \\  1, & \text{ if }x \in  {\Omega }_{0}, \end{array}\right. \tag{7}
$$

where $\gamma$ is the geometric parameter to determine the boundary curve function ${F}_{\gamma }\left( x\right)  = 0$ for $x$ at the boundary $\Gamma , n$ is the number of patches (different discrete topology patches as illustrated in Fig. 2(a)), and $\beta$ is a parameter to determine the sharpness of $\delta$ . SDF is the signed distance function [43], and its sign is determined by the relative location of a sample point to the topology. If a sample point is inside the topology $\left( {x \in  {\Omega }_{1}}\right)$ , its sign is -1 ; if it is on the topology boundary $\left( {x \in  \partial {\Omega }_{1}}\right)$ , the sign is 0 ; if it is outside the topology $\left( {x \in  {\Omega }_{0}}\right)$ , the sign is $1.d\left( {x,{F}_{\gamma }}\right)$ is the distance between a sample point and its nearest boundary point, which can be written as:

$$
d\left( {\mathbf{x},{F}_{\gamma }}\right)  = {\begin{Vmatrix}\mathbf{x} - {\mathbf{x}}_{\partial {\Omega }_{1}}^{ * }\end{Vmatrix}}_{2}, \tag{8}
$$

where ${\mathbf{x}}_{\partial {\Omega }_{1}}^{ * }$ is the nearest boundary point to a given sample point.

Compared with DT-PINNs that rely on implicit density $\left( \rho \right)$ to represent the topology, LT-PINNs introduce the explicit distance function $\left( \delta \right)$ as topology representation, overcoming the diffuse boundary limitations in DT-PINNs. For systems with multiple topology patches, each patch is characterized by its own geometric distance function, and their collective influence on Eq. (6) is mediated through the cumulative product of these functions. Taking a circle topology patch with a unit diameter for example (as shown in Fig. 3), its center is denoted by $\gamma$ . The boundary curve function for this patch is given by ${F}_{\gamma }\left( \mathbf{x}\right)  = {\left( \mathbf{x} - \gamma \right) }^{2} - {0.5}^{2} = 0$ , and the corresponding geometric distance function is $\delta \left( {\mathbf{x},\mathbf{\gamma }}\right)  = \frac{1}{1 + {e}^{-\beta  \cdot  \left( {\parallel \mathbf{x} - \mathbf{\gamma }{\parallel }_{2} - {0.5}}\right) }}$ . On the boundary, $\delta  = {0.5}$ ; inside the patch, $\delta$ decays to 0 quickly; outside the patch, $\delta$ increases to 1 rapidly. The geometric distance function is only determined by the geometric parameter, thus it results in a more distinct boundary and a more efficient topology representation compared with implicit density representation used in DT-PINNs.

![5_501_167_693_613_0.jpg](images/5_501_167_693_613_0.jpg)

Fig. 3. An example of geometric distance function with a unit diameter circle topology patch.

A comparison of different topology representation approaches, including the density approach used in DT-PINNs, the Lagrangian approach in LT-PINNs, and Fourier approach in FF-PINNTO [26], is made on simple 1-dimensional (1D) step geometry and 2- dimensional (2D) circle geometry. Although the level-set approach is important, we refer its performance to the density approach duo to their inherent correlation mentioned above. The results are described in Appendix B.

In addition to Eq. (6), LT-PINNs impose directly boundary condition constraints, as illustrated in Fig. 2(b). In general, $\mathcal{B}$ denotes any boundary conditions operator and $g$ is a known function. For Dirichlet boundary conditions, the boundary conditions operator is $\mathcal{B} : \mathbf{u} = {\mathbf{u}}_{b}^{ * }$ . Dirichlet boundary conditions can be enforced without geometric information of the boundary, whereas Neumann boundary conditions depend on the boundary normal vector, represented mathematically as:

$$
\mathbf{n} = \frac{\nabla {F}_{\gamma }}{{\begin{Vmatrix}\nabla {F}_{\gamma }\end{Vmatrix}}_{2}}, \tag{9}
$$

and the Neumann boundary condition operator is:

$$
\mathcal{B} : \frac{\partial \mathbf{u}}{\partial \mathbf{n}} = \nabla \mathbf{u} \cdot  \frac{\nabla {F}_{\gamma }}{{\begin{Vmatrix}\nabla {F}_{\gamma }\end{Vmatrix}}_{2}} = q, \tag{10}
$$

where $q$ is the flux across the boundary. Therefore, the PDE, boundary condition, and data loss functions for LT-PINNs are described as:

$$
{L}_{p} = \frac{1}{N}\mathop{\sum }\limits_{{i = 1}}^{N}{R}_{i}^{2},{R}_{i} = {\left( \mathop{\prod }\limits_{1}^{n}{\delta }^{k}\left( \mathbf{x},\mathbf{\gamma }\right) \right) }_{i}{\left( {\mathcal{N}}_{{\omega }^{ * }}\left\lbrack  \mathbf{u}\left( \mathbf{x}, t\right) \right\rbrack   - f\left( \mathbf{x}, t\right) \right) }_{i},
$$

$$
{L}_{b} = \left\{  \begin{array}{ll} \frac{1}{{N}_{b}}\mathop{\sum }\limits_{{i = 1}}^{{N}_{b}}{\left( \mathbf{u} - {u}_{b}^{ * }\right) }_{i}^{2}, & \text{ if Dirichlet b.c., } \\  \frac{1}{{N}_{b}}\mathop{\sum }\limits_{{i = 1}}^{{N}_{b}}{\left( \nabla \mathbf{u} \cdot  \frac{\nabla {F}_{\gamma }}{{\begin{Vmatrix}\nabla {F}_{\gamma }\end{Vmatrix}}_{2}} - q\right) }_{i}^{2}, & \text{ if Neumann b.c., } \end{array}\right. \tag{11}
$$

$$
{L}_{d} = \frac{1}{{N}_{d}}\mathop{\sum }\limits_{{i = 1}}^{{N}_{d}}{\left( \mathbf{u} - {\mathbf{u}}^{ * }\right) }_{i}^{2}
$$

where ${L}_{p}$ is the PDE loss, ${L}_{b}$ is the boundary condition loss, and ${L}_{d}$ is the data loss, $R$ is the residual of PDE, ${\mathbf{u}}_{b}^{ * }$ are Dirichlet boundary conditions, ${u}^{ * }$ are the sparse measurements/initial conditions on $\mathbf{u}, N$ is the number of sample points for PDE loss, ${N}_{b}$ is the number of sample points for boundary condition loss, and ${N}_{d}$ is the number of sample points for data loss.

Since DT-PINNs lack explicit topological representation, they can only enforce Dirichlet boundary conditions indirectly through the composite PDE loss in Eq. (5). This leads to imprecise boundary constraint implementation. Conversely, LT-PINNs employ a dedicated boundary condition loss function (Eq. (11)) that essentially supports both Dirichlet and Neumann conditions, which also has the potentiality to be applied to arbitrary boundary conditions, yielding precise boundary constraints. This enhanced loss functions enable more accurate PDE solutions and straightforward topology inference without manual interpolation.

Additionally, since most complex topology systems exhibit self-similar patterns and multi-scale hierarchical structures (e.g., lattice-based metamaterials, fractal geometries, or bio-inspired cellular architectures), DT-PINNs' implicit density representation struggles to capture these repetitive features efficiently. In contrast, LT-PINNs' explicit geometric distance functions can naturally encode such periodicity and pattern recurrence through two types of topology loss functions $\left( {{L}_{t}^{1},{L}_{t}^{2}}\right)$ . They are formulated based on the distance between pairs of topology patches, defined as:

$$
{L}_{t}^{1} = \frac{1}{N}\mathop{\sum }\limits_{{i = 1}}^{{N}_{t}}\mathop{\sum }\limits_{{j \neq  i, j = 1}}^{{N}_{t}}{\left( {r}_{ij} - {r}_{ij}^{ * }\right) }^{2}, \tag{12}
$$

$$
{L}_{t}^{2} = \frac{1}{N}\mathop{\sum }\limits_{{i = 1}}^{N}\mathop{\sum }\limits_{{j \neq  i, j = 1}}^{{N}_{t}}\frac{1}{{r}_{ij}^{2}},
$$

where ${N}_{t}$ is the number of topology patches, ${r}_{ij}$ is the distance between a pair of topology patches, ${r}_{ij}^{ * }$ is the reference distance, ${L}_{t}^{1}$ is the loss related to a fixed distance constraint, and ${L}_{t}^{2}$ is the loss related to non-overlapping constraint. Therefore, the total loss function of LT-PINNs that incorporates aforementioned topology loss function is:

$$
{L}_{tot}\left( {\theta ,\gamma }\right)  = {\lambda }_{p}{L}_{p} + {\lambda }_{b}{L}_{b} + {\lambda }_{d}{L}_{d} + {\lambda }_{t}{L}_{t}, \tag{13}
$$

where ${L}_{tot}$ is the total loss, $\theta$ and $\gamma$ represent the parameters in the LT-PINN, ${\lambda }_{t}$ is the weight of ${L}_{t}$ , which is sum of ${L}_{t}^{1}$ and ${L}_{t}^{2},{\lambda }_{p},{\lambda }_{b}$ and ${\lambda }_{d}$ are the weights of the PDE, boundary condition, and data loss, respectively.

The total loss minimization follows the stochastic gradient descent. In particular, for PDEs with Neumann boundary conditions on a single topology patch, $\gamma$ is updated via backpropagation according to the gradient of the corresponding loss:

$$
{\gamma }^{l + 1} = {\gamma }^{l} - \eta \left( {{\lambda }_{p}\frac{\partial {L}_{p}}{\partial \gamma } + {\lambda }_{b}\frac{\partial {L}_{b}}{\partial \gamma }}\right)
$$

$$
\frac{\partial {L}_{p}}{\partial \mathbf{\gamma }} = \frac{1}{N}\mathop{\sum }\limits_{{i = 1}}^{N}{\left( \mathop{\prod }\limits_{1}^{n}\frac{\partial {\left( {\delta }^{k}\left( \mathbf{x},\mathbf{\gamma }\right) \right) }^{2}}{\partial \mathbf{\gamma }}\right) }_{i}{\left( {\mathcal{N}}_{{\omega }^{ * }}\left\lbrack  \mathbf{u}\left( \mathbf{x}, t\right) \right\rbrack   - f\left( \mathbf{x}, t\right) \right) }_{i}^{2}, \tag{14}
$$

$$
\frac{\partial {L}_{b}}{\partial \gamma } = \frac{2}{{N}_{b}}\mathop{\sum }\limits_{{i = 1}}^{{N}_{b}}{\left( \nabla \mathbf{u} \cdot  \frac{\nabla {F}_{\gamma }}{{\begin{Vmatrix}\nabla {F}_{\gamma }\end{Vmatrix}}_{2}} - q\right) }_{i}{\left( \nabla \mathbf{u} \cdot  \frac{\partial \nabla {F}_{\gamma }}{\partial \gamma {\begin{Vmatrix}\nabla {F}_{\gamma }\end{Vmatrix}}_{2}}\right) }_{i},
$$

where $l$ is the training step and $\eta$ is the learning rate. Since $\frac{\partial {\left( {\delta }^{k}\left( \mathbf{x},\mathbf{y}\right) \right) }^{2}}{\partial \mathbf{\gamma }}$ and $\frac{\partial \nabla {F}_{\gamma }}{\partial \mathbf{\gamma }{\begin{Vmatrix}\nabla {F}_{\gamma }\end{Vmatrix}}_{2}}$ are rarely zero, it ensures $\mathbf{\gamma }$ converges to yield solutions fulfilling the PDEs and boundary conditions.

The major difference between LT-PINNs and DT-PINNs lies in the parameterization of $\gamma$ as learnable variables in LT-PINNs. This design enables the topology boundary curve function ${F}_{\gamma }\left( x\right)  = 0$ to dynamically evolve during training, offering three key advantages: (1) elimination of error-prone manual boundary interpolation required in DT-PINNs, (2) precise boundary constraints for both Dirichlet and Neumann boundary conditions, and (3) naturally encoding periodicity and pattern recurrence in complex topology systems. However, LT-PINNs require prior knowledge about ${F}_{\gamma }$ .

For further investigation, soft or hard geometric constraints can be embedded into DT-PINNs, such as periodic boundary conditions, or more smooth and differentiable topology representations achieved through Fourier mapping and periodic activation functions [26]. Extending this idea, when complete geometric information is available and hard constraints are fully enforced, DT-PINNs naturally transition into LT-PINNs. In this sense, DT-PINNs and LT-PINNs can be viewed as two ends of the topology optimization spectrum: DT-PINNs emphasize flexibility and adaptability under minimal prior knowledge of topology, whereas LT-PINNs prioritize accuracy and manufacturability when sufficient prior knowledge of topology is provided.

## 3. Numerical experiments

While topology optimization often produces complex designs, practical applications frequently prioritize simple, manufacturable geometries. For instance, a simple pipe is often preferred over a complex fluid channel for many engineering purposes [29]. Therefore, it is essential to evaluate LT-PINNs on both simple and complex design tasks. Accordingly, this study begins with simple primitive design tasks and gradually increases their complexity. A series of numerical experiments for simple primitive topology optimization are first performed to evaluate the accuracy of LT-PINNs against DT-PINNs. These experiments involve various PDEs governed by both Dirichlet and Neumann boundary conditions. Moreover, we demonstrate the applicability of LT-PINNs in solving more complex topology optimization problems, involving multi-circle array topology for both steady and unsteady flow scenarios. Finally, we test LT-PINNs on flow rearrangement tasks, considering cases without prior known measurement data. For simplicity, all simulations are restricted to two-dimensional domains and all topology patches have a characteristic length of 1 . For example, the boundary curve function of the single circle in Eq. (11) is ${F}_{\gamma }\left( \mathbf{x}\right)  = {\left( \mathbf{x} - \gamma \right) }^{2} - {0.5}^{2} = 0$ , where $\gamma$ means the adaptive center of circle. Consequently, the geometric distance function defined in Eq. (7) in Section 2.2 is $\delta \left( {\mathbf{x},\mathbf{\gamma }}\right)  = \frac{1}{1 + {e}^{-\beta  \cdot  \left( {\parallel \mathbf{x} - \mathbf{\gamma }{\parallel }_{2} - {0.5}}\right) }}$ , and the normal of the boundary curve is $\mathbf{n} = 2\left( {\mathbf{x} - \mathbf{\gamma }}\right)$ . We choose $\beta  = {100}$ for the geometric distance function to ensure that $\mathop{\prod }\limits_{1}^{n}{\delta }^{k}\left( {\mathbf{x},\mathbf{\gamma }}\right)$ has a sharp conversion between 0 and 1. In addition, all trainable parameters $\theta$ in PINNs are initialized by He initialization [44], while the $\gamma$ is randomly initialized firstly and then normalized using the sigmoid function to fully span within the ROI. As a result, the optimization of $\gamma$ remains constrained to this predefined ROI spatial domain, avoiding unnecessary computational overhead.

All the reference solution data is generated by Star-CCM + 2021.1 and all the training of PINN models are performed on the NVIDIA RTX 4090 GPUs with implementations using the framework PyTorch 2.1.0. The details of the computational setup, e.g., architectures of neural networks, optimizer, and so on, for all test cases are present in Appendix A in addition to the cases in Sections 3.1 and 3.2. It should be noted that the computational model setup is tailored for each specific case to achieve optimal results. Data and code are available at: https://github.com/cloud2009/LT-PINN.

Table 1

Elastic equation: details of the PINN architectures and operating parameters.

<table><tr><td>Settings</td><td>DT-PINN</td><td>LT-PINN</td></tr><tr><td>#of layers</td><td>5</td><td>5</td></tr><tr><td>#of neurons per layer</td><td>64</td><td>64</td></tr><tr><td>activation function</td><td>Tanh</td><td>Tanh</td></tr><tr><td>optimizer</td><td>Adam</td><td>Adam</td></tr><tr><td>learning rate</td><td>$1 \times  {10}^{-4}$</td><td>$1 \times  {10}^{-4}$</td></tr><tr><td>training epoch</td><td>400,000</td><td>400,000</td></tr><tr><td>$N$</td><td>120×120</td><td>120×120</td></tr><tr><td>${N}_{d}$</td><td>2167</td><td>2167</td></tr><tr><td>${N}_{b}$</td><td>-</td><td>512</td></tr><tr><td>${N}_{t}$</td><td>-</td><td>-</td></tr><tr><td>${\lambda }_{p}$</td><td>$2 \times  {10}^{2}$</td><td>$2 \times  {10}^{3}$</td></tr><tr><td>${\lambda }_{d}$</td><td>$1 \times  {10}^{4}$</td><td>$1 \times  {10}^{4}$</td></tr><tr><td>${\lambda }_{b}$</td><td>-</td><td>$1 \times  {10}^{4}$</td></tr><tr><td>${\lambda }_{t}$</td><td>-</td><td>-</td></tr><tr><td>PDE</td><td>Eq. (15)</td><td>Eq. (15)</td></tr><tr><td>#of topology patches</td><td>-</td><td>1</td></tr><tr><td>#of GPUs</td><td>1</td><td>1</td></tr></table>

### 3.1. Benchmarking and comparative analysis

In this section, we compare DT-PINNs and LT-PINNs in solving for the solution $u$ and inferring topology, focusing on the elastic equations with Dirichlet boundary conditions and Laplace's equations with Neumann boundary conditions. Since these two types of boundary conditions serve as the foundation for many others, such as Robin, Cauchy, and symmetry boundary conditions, they are a reasonable choice for test cases. For elastic equations, the Dirichlet boundary conditions, such as the fixed wall boundary, is common in structure analysis. For Laplace's equations, the Neumann boundary conditions, such as constant normal derivative of the solution on the boundary, are also commonly used as illustrative examples. These two types of equations contain second order partial differential terms that are able to represent general features of PDEs.

The training data for the PINNs are generated within the computational domain illustrated in Fig. 4(a). Note that ${\Omega }_{1}$ represents the circle topology domain and ${\Omega }_{0}$ represents the rest domain for surrounding medium. The diameter of circle is set to 1, i.e., $D = 1$ . The full domain spans ${15D} \times  {25D}$ , while the training data for the PINNs are extracted from a local ROI of ${2.2D} \times  {2.2D}$ centered around the circle. Within the ROI, sample points are uniformly distributed inside the ${2D} \times  {2D}$ region covering the topological feature, while outside this region, they are randomly distributed, as illustrated in Fig. 4(b). Boundary conditions on all four edges of ROI are prescribed according to the generated data. Given the sparse measurement data in the ROI, where the topology is unknown, PINNs are employed to simultaneously determine the solution $u$ and identify the topology.

#### 3.1.1. Test case for elastic equation with dirichlet boundary conditions

We first test the PINNs on the elastic equation with Dirichlet boundary conditions. The elastic equation is defined as:

$$
\nabla  \cdot  \mathbf{\sigma } = 0
$$

$$
\epsilon  = \frac{1}{2}\left( {\nabla \mathbf{u} + {\left( \nabla \mathbf{u}\right) }^{T}}\right) \tag{15}
$$

$$
{\left. u\right| }_{bc} = 0
$$

where $\sigma$ is the normalized stress tensor based on reference stress ${\sigma }_{ref}$ . The three stress components are $\left\lbrack  {{\sigma }_{xx},{\sigma }_{yy},{\sigma }_{xy}}\right\rbrack  .\epsilon$ is the normalized strain tensor based on reference length ${u}_{ref}$ . The three strain components are $\left\lbrack  {{\epsilon }_{xx},{\epsilon }_{yy},{\epsilon }_{xy}}\right\rbrack  .\mathbf{u}$ is the normalized displacement based on reference length ${u}_{ref}$ , and it has two components, i.e., $\left\lbrack  {{u}_{x},{u}_{x}}\right\rbrack$ .

In this study, a linear elastic, isotropic material is considered, which is governed by the following constitutive equation:

$$
{\sigma }_{xx} = \frac{E}{1 - {v}^{2}}\left( {{\epsilon }_{xx} + v{\epsilon }_{yy}}\right)
$$

$$
{\sigma }_{yy} = \frac{E}{1 - {v}^{2}}\left( {{\epsilon }_{yy} + v{\epsilon }_{xx}}\right) \tag{16}
$$

$$
{\sigma }_{xy} = \frac{E}{1 - v}{\epsilon }_{xy}
$$

where $E$ is Young’s module based on the reference strain ${\sigma }_{ref}$ , and $v$ is Poisson ratio. The relation between the strain and displacement is:

$$
{\epsilon }_{xx} = \frac{\partial {u}_{x}}{\partial x}
$$

$$
{\epsilon }_{yy} = \frac{\partial {u}_{y}}{\partial y} \tag{17}
$$

$$
{\epsilon }_{xy} = \frac{1}{2}\left( {\frac{\partial {u}_{x}}{\partial y} + \frac{\partial {u}_{y}}{\partial x}}\right) .
$$

![8_472_161_758_1061_0.jpg](images/8_472_161_758_1061_0.jpg)

Fig. 4. Illustration of computational domain and data sampling strategy: a) the whole simulation domain; b) the data sampling strategy in ROI.

To generate data for training, we set $E = 1$ and $v = {0.33}$ . It is noted that all physical parameters are normalized in this study, to maintain non-dimensional forms, allowing us to focus on the model's general performance. The edge conditions applied on four edges of the entire domain are specified as follows: free moving (top and bottom edges), fixed (right edge), and uniform pressure of 1 (left edge). The random distributed data points outside ${2D} \times  {2D}$ domain are used as sparse measurement data, while the uniformly distributed data points inside region ${2D} \times  {2D}$ are used to calculate PDE loss. For LT-PINN, the boundary points are uniformly sampled along four concentric rings within the circle topology with different radii at0.5,0.4,0.3, and 0.2 . On each concentric ring, the number of sampled boundary points is 128. Detailed description of data preparation and training settings for PINNs are listed in Table 1.

In the DT-PINN, the largest mono-convex topology is obtained by manually thresholding the predicted $\widehat{\rho }$ to define the boundary interface. LT-PINN, however, eliminates this step by directly deriving the topology boundary from the learned curve function ${F}_{\gamma }\left( x\right)  =$ 0, resulting in an automated and computationally efficient approach. The predictive performance of DT-PINN versus LT-PINN is compared in Fig. 5, evaluating both the displacement solution $u$ and identified topological boundary in the ${2D} \times  {2D}$ region. The results demonstrate that LT-PINN accurately locates the circle patch, showing close agreement with the reference topology. In contrast, DT-PINN infers a small elliptical topology, whose center is offset to the left of the reference circle patch. Furthermore, comparison of the displacement fields reveals that LT-PINN achieves better agreement with the reference solution than DT-PINN. Notably, for the small $y$ -direction displacement prediction shown in Fig 5(c), the LT-PINN also predicts more accurate displacement isolines than DT-PINN.

Fig. 6 compares the predicted displacements along the lines $x = 0$ and $y = 0$ with the reference solution. The LT-PINN results demonstrate closer agreement with the reference data for all components ( $x$ -direction, $y$ -direction, and magnitude), consistent with the observations in Fig. 5. Clearly, while the reference $y$ -displacement along $y = 0$ exhibits fluctuations due to small numerical interpolation artifacts, LT-PINN produces a physically more plausible smooth profile. These results indicate that LT-PINN not only accurately identifies the unknown topology, but also demonstrates robustness against small numerical artifacts (i.e., interpolation error) in the reference data.

![9_235_157_1239_1189_0.jpg](images/9_235_157_1239_1189_0.jpg)

Fig. 5. Elastic equation: predicted displacement and topology boundary via different PINNs: a) the displacement magnitude; b) $x$ -direction displacement; c) $y$ -direction displacement.

Table 2

Elastic equation: relative ${L}_{2}$ error of predicted displacement via different PINNs.

<table><tr><td>Relative ${L}_{2}$ error</td><td>DT-PINN</td><td>LT-PINN</td></tr><tr><td>$x$ -displacement</td><td>0.0998</td><td>0.0334</td></tr><tr><td>$y$ -displacement</td><td>0.0875</td><td>0.0696</td></tr><tr><td>Displacement magnitude</td><td>0.0709</td><td>0.0238</td></tr></table>

Moreover, the prediction error is quantified using the relative ${L}_{2}$ error, evaluated on a ${128} \times  {128}$ uniform grid of test sample points within the ${2D} \times  {2D}$ region (excluding points inside the circle). The corresponding error metrics are presented in Table 2. LT-PINN demonstrates superior accuracy with significantly lower relative ${L}_{2}$ errors compared to DT-PINN. Specifically, the relative ${L}_{2}$ errors are reduced by 66.53% for $x$ -direction displacement,20.46% for $y$ -direction displacement, and 66.43% for displacement magnitude. These substantial improvements confirm that LT-PINN provides more accurate predictions for both the topological features and displacement fields. Based on the comprehensive analysis presented above, it is convinced that, for topology optimization governed by the elastic equation with Dirichlet boundary conditions, LT-PINN outperforms DT-PINN in both solving the governing PDEs and inferring topology.

![10_312_162_1083_1508_0.jpg](images/10_312_162_1083_1508_0.jpg)

Fig. 6. Elastic equation: predicted displacement along the lines $x = 0$ and $y = 0$ : a) the displacement magnitude; b) $x$ -direction displacement; c) $y$ -direction displacement.

#### 3.1.2. Test case for Laplace's equation with Neumann boundary conditions

We then use Laplace's equation with Neumann boundary conditions to test PINNs. In this case, the Laplace's equation is:

$$
{\nabla }^{2}T = 0
$$

$$
{\left. \nabla T \cdot  \mathbf{n}\right| }_{bc} = q, \tag{18}
$$

where $T$ is temperature, $q$ is the heat flux across boundary, and $n$ is the normal of boundary. It is noted that both $T$ and $q$ are normalized for simplicity.

To generate data for training, we set $q =  - {0.5}$ . The boundary normal’s positive direction is defined as pointing the outward from inner circular center. A negative heat flux indicates that the temperature gradient vector opposes the boundary normal direction, implying that the boundary temperature is higher than the exterior. As a result, the circle can be treated as a heat source. The boundary conditions applied on the four edges of the entire domain are all constant with $T = 0$ . Training is performed exclusively within the ${2D} \times  {2D}$ region, using a ${120} \times  {120}$ uniform grid of sample points for PDE loss evaluation and 1638 randomly sampled points from this grid as sparse measurement data points of the solution $T$ . This formulation treats topology optimization as a high-resolution reconstruction problem from low-resolution measurement data. For LT-PINN, the boundary points are uniformly sampled along four concentric rings within the circle topology with different radii at 0.5, 0.4, 0.3, and 0.2 . On each concentric ring, the number of sampled boundary points is 256. Detailed description of data preparation and training settings for PINNs are listed in Table 3.

Table 3

Laplace's equation: details of the PINN architectures and operating parameters.

<table><tr><td>Settings</td><td>DT-PINN</td><td>LT-PINN</td></tr><tr><td>#of layers</td><td>8</td><td>8</td></tr><tr><td>#of neurons per layer</td><td>256</td><td>256</td></tr><tr><td>activation function</td><td>Tanh</td><td>Tanh</td></tr><tr><td>optimizer</td><td>Adam</td><td>Adam</td></tr><tr><td>learning rate</td><td>$1 \times  {10}^{-4}$</td><td>$1 \times  {10}^{-4}$</td></tr><tr><td>training epoch</td><td>400,000</td><td>400,000</td></tr><tr><td>$N$</td><td>${120} \times  {120}$</td><td>120×120</td></tr><tr><td>${N}_{d}$</td><td>1,638</td><td>1,638</td></tr><tr><td>${N}_{b}$</td><td>-</td><td>1,024</td></tr><tr><td>${N}_{t}$</td><td>-</td><td>-</td></tr><tr><td>${\lambda }_{p}$</td><td>1</td><td>1</td></tr><tr><td>${\lambda }_{d}$</td><td>$1 \times  {10}^{4}$</td><td>$1 \times  {10}^{4}$</td></tr><tr><td>${\lambda }_{b}$</td><td>-</td><td>1</td></tr><tr><td>${\lambda }_{t}$</td><td>-</td><td>-</td></tr><tr><td>PDE</td><td>Eq. (18)</td><td>Eq. (18)</td></tr><tr><td>#of topology patches</td><td>-</td><td>1</td></tr><tr><td>#of GPUs</td><td>1</td><td>1</td></tr></table>

![11_281_817_1147_291_0.jpg](images/11_281_817_1147_291_0.jpg)

Fig. 7. Laplace's equation: reconstructed temperature field and topology boundary via different PINNs.

As $\widehat{\rho }$ is the density representation of topology in DT-PINNs (see Eq. (5)), a threshold value of the predicted $\widehat{\rho }$ is needed to separate topology from its surrounding domain. It is manually selected to ensure the largest mono-convex topology is distinguished in DT-PINNs. However, for LT-PINNs, it does not require manual interpolation to extract topology. In LT-PINNs, the optimized parameter $\gamma$ and the boundary function ${F\gamma }\left( \mathbf{x}\right)  = 0$ explicitly define the topology, offering greater reliability and efficiency compared to DT-PINNs.

The resulting reconstructed temperature $T$ and topology boundary in region ${2D} \times  {2D}$ are shown in Fig. 7. The results demonstrate that LT-PINN accurately identifies both the position and shape of the circle, showing close agreement with the reference solution. In contrast, DT-PINN produces an irregular topology with prominent spikes. Furthermore, temperature field reconstruction reveals LT-PINN's enhanced prediction accuracy. DT-PINN, on the other hand, generates less smooth temperature field characterized by circumferential fluctuations in the isolines, which is directly correlated with DT-PINN's irregular topology boundary.

The temperature profiles along two lines $x = 0$ and $y = 0$ are also analyzed, with zoom-in views of the boundary regions extracted and shown in Fig. 8. The temperature profiles along both lines show minimal discrepancies between the reference data and both models' predictions, given low-resolution reference temperature as measurement data for training. A minor discrepancy of the temperature profile along the line $x = 0$ (Fig. 8a) from DT-PINN is observed around $y =  - {0.5}$ , due to the irregularly reconstructed topology. However, a closer examination of the boundary regions (Fig. 8b) reveals noticeable deviations between the reference solution and DT-PINN's prediction, while LT-PINN's reconstructed temperature profiles maintain better agreement with the reference data.

The heat flux (i.e., normal gradient of temperature) across the reference boundary is also present in Fig. 9. Obviously, DT-PINN fails to predict a constant heat flux across the boundary defined by the Neumann boundary conditions (i.e., $q =  - {0.5}$ ), showing significant deviations from the reference. It tends to predict a smaller and random heat flux, i.e., a smaller temperature gradient on the boundary, which is supposed to be a sign of spectral bias of PINNs [45,46]. Conversely, LT-PINN predicts almost the same constant heat flux that closely matches the reference.

The accuracy of the reconstructed temperature field is quantified using the relative ${L}_{2}$ error, evaluated on a ${128} \times  {128}$ uniform grid of test sample points within the ${2D} \times  {2D}$ domain (excluding points inside the circle). Moreover, the error in the heat flux across the boundary is computed. Both error metrics are summarized in Table 4. Despite using identical low-resolution temperature data in the data loss function for both methods, LT-PINN achieves significantly lower errors: a 44.90% reduction in the relative ${L}_{2}$ error for the temperature field and a 99.42% reduction for the boundary heat flux, compared to DT-PINN. This disparity arises because DT-PINN fails to enforce Neumann boundary conditions in its total loss function, resulting in inaccurate boundary predictions and degraded temperature field reconstruction. Conversely, LT-PINN explicitly incorporates these boundary conditions, leading to improved accuracy. Therefore, for topology optimization governed by Laplace's equation with Neumann boundary conditions, LT-PINN excels DT-PINN in predictive accuracy for both the PDE solution and topology. This advantage stems primarily from LT-PINN's rigorous incorporation of Neumann boundary conditions into its total loss function.

![12_334_160_1040_1065_0.jpg](images/12_334_160_1040_1065_0.jpg)

Fig. 8. Laplace’s equation: reconstructed temperature along $x = 0$ and $y = 0$ lines: a) temperature along the whole line; b) zoom-in view of temperature near boundary.

Table 4

Laplace’s equation: relative ${L}_{2}$ error of reconstructed temperature and heat flux via different PINNs.

<table><tr><td>Relative ${L}_{2}$ error</td><td>DT-PINN</td><td>LT-PINN</td></tr><tr><td>Temperature</td><td>0.0049</td><td>0.0027</td></tr><tr><td>Heat flux across boundary</td><td>0.4006</td><td>0.0023</td></tr></table>

### 3.2. Implementing LT-PINNs for different primitive topology optimization

The capabilities of LT-PINNs have been demonstrated in Section 3.1 with simple circle geometry through comparative benchmarking against DT-PINNs. Given these promising results, we now explore LT-PINNs' potential in tackling a broader range of primitive geometric configurations, including triangles with varying angles, rectangles with varying aspect ratios, and bricks with varying aspect ratios, for further performance evaluation.

The Laplace's equation with Dirichlet boundary conditions is specifically used for the experiment. The Laplace's equation and its boundary condition are:

$$
{\nabla }^{2}T = 0
$$

$$
{\left. T\right| }_{bc} = 1 \tag{19}
$$

![13_506_162_691_574_0.jpg](images/13_506_162_691_574_0.jpg)

Fig. 9. Laplace's equation: reconstructed heat flux across the reference boundary via different PINNs.

![13_248_828_1213_443_0.jpg](images/13_248_828_1213_443_0.jpg)

Fig. 10. Configurations of different primitive topologies: a) triangular with varying angles; b) rectangular with varying aspect ratios; c) concave brick with varying aspect ratios.

Table 5

Parameter settings of generated primitive topologies.

<table><tr><td rowspan="2"></td><td colspan="3">Triangle</td><td colspan="3">Rectangle</td><td colspan="3">Brick</td></tr><tr><td>$\alpha$</td><td>${x}_{0}$</td><td>${y}_{0}$</td><td>$L$</td><td>${x}_{0}$</td><td>${y}_{0}$</td><td>$L$</td><td>${x}_{0}$</td><td>${y}_{0}$</td></tr><tr><td>Setting A</td><td>60.0°</td><td>0.0</td><td>0.0</td><td>0.5</td><td>0.0</td><td>0.0</td><td>0.5</td><td>0.0</td><td>0.0</td></tr><tr><td>Setting B</td><td>120.0°</td><td>0.0</td><td>0.0</td><td>1.5</td><td>0.0</td><td>0.0</td><td>1.5</td><td>0.0</td><td>0.0</td></tr></table>

where $T$ is the temperature and ${\left. T\right| }_{bc}$ is the temperature on the boundary. They are normalized from 0 to 1 for simplicity.

The size of the computing domain used for generating the data is the same as that described in Section 3.1.2, and the four edges of the entire domain have constant temperature $T = 0$ . The topology is replaced by the new geometric configurations in the ROI, where a portion of the data is extracted for training, as illustrated in Fig. 10. The triangle and rectangle represent the convex shape, and the brick represents the concave shape. For simplicity, the triangle is isosceles triangle with a length of 1, the heights (H) of the both rectangle and brick are 1, and the notch of the brick is $h = {0.5}\mathrm{H}$ in height and $l = {0.5}\mathrm{\;L}$ in length. The size of the ROI for triangle, rectangle, and brick is 2 × 2. Two sets of geometric parameters, including the apex angle of the triangle ( $\alpha$ ), the length of the rectangle and brick $\left( L\right)$ , and their centers $\left( {{x}_{0},{y}_{0}}\right)$ are generated for this case study, as listed in Table 5.

In order to describe the topology boundary curve function $\left( {{F\gamma }\left( \mathbf{x}\right)  = 0}\right)$ for these primitive geometric configurations, piece-wise lines are introduced. For triangle, the topology boundary curve function is defined as:

$$
\left\{  \begin{array}{ll} y = \frac{2}{3}\cos \left( \frac{\alpha }{2}\right)  + \cot \left( \frac{\alpha }{2}\right) x, & x \in  \left\lbrack  {-\sin \frac{\alpha }{2},0}\right\rbrack  , \\  y = \frac{2}{3}\cos \left( \frac{\alpha }{2}\right)  - \cot \left( \frac{\alpha }{2}\right) x, & x \in  \left\lbrack  {0,\sin \frac{\alpha }{2}}\right\rbrack  , \\  y =  - \frac{1}{3}\cos \left( \frac{\alpha }{2}\right) , & x \in  \left\lbrack  {-\sin \frac{\alpha }{2},\sin \frac{\alpha }{2}}\right\rbrack  . \end{array}\right. \tag{20}
$$

For rectangle, it is defined as:

$$
\left\{  \begin{array}{ll} x = \frac{L}{2}, & y \in  \left\lbrack  {-\frac{H}{2},\frac{H}{2}}\right\rbrack  , \\  x =  - \frac{L}{2}, & y \in  \left\lbrack  {-\frac{H}{2},\frac{H}{2}}\right\rbrack  , \\  y = \frac{H}{2}, & x \in  \left\lbrack  {-\frac{L}{2},\frac{L}{2}}\right\rbrack  , \\  y =  - \frac{H}{2}, & x \in  \left\lbrack  {-\frac{L}{2},\frac{L}{2}}\right\rbrack  . \end{array}\right. \tag{21}
$$

For brick, it is defined as:

$$
\left\{  \begin{array}{ll} x =  - \frac{L}{2}, & y \in  \left\lbrack  {-\frac{H}{2},\frac{H}{2}}\right\rbrack  , \\  y = \frac{H}{2}, & x \in  \left\lbrack  {-\frac{L}{2}, - \frac{1}{2}}\right\rbrack  , \\  x =  - \frac{1}{2}, & y \in  \left\lbrack  {\frac{H}{2} - h,\frac{H}{2}}\right\rbrack  , \\  y = \frac{H}{2} - h, & x \in  \left\lbrack  {-\frac{L}{2},\frac{1}{2}}\right\rbrack  , \\  x = \frac{L}{2}, & y \in  \left\lbrack  {\frac{H}{2} - h,\frac{H}{2}}\right\rbrack  , \\  y = \frac{H}{2}, & x \in  \left\lbrack  {\frac{L}{2},\frac{L}{2}}\right\rbrack  , \\  x = \frac{L}{2}, & y \in  \left\lbrack  {-\frac{L}{2},\frac{H}{2}}\right\rbrack  , \\  y =  - \frac{H}{2}, & x \in  \left\lbrack  {-\frac{L}{2},\frac{L}{2}}\right\rbrack  , \end{array}\right. \tag{22}
$$

The minimal surface distance from a point to a piecewise-line defines its distance to the geometry. Classifying a point as inside or outside is determined by its position relative to these lines. For a convex shape (e.g., a triangle or rectangle), a point is inside if it lies on the same side of all boundary segments. For a concave shape (e.g. brick), a different approach is needed. Here, the shape is decomposed into convex primitives. For instance, the brick can be divided into a large rectangle $\left( {H \times  L}\right)$ and a smaller notch rectangle $\left( {h \times  l}\right)$ . A point is then inside the overall geometry if it is inside the large rectangle and outside the small notch rectangle.

The learnable geometric parameters $\left( \gamma \right)$ of LT-PINNs include the those listed in Table 5. The boundary curve function $\left( {{F\gamma }\left( \mathbf{x}\right)  = 0}\right)$ is constructed from piece-wise lines, with each line transposed according to its relative position with respect to the center. Training is conducted exclusively within the ROI using a uniform grid. A unit density of 60 is used for PDE loss evaluation, and from this grid,100 points are randomly sampled to serve as sparse measurement data for the solution $T$ . These measurement points constitute less than $1\%$ of the total grid, framing this as a challenging high-resolution reconstruction problem from ultra-sparse data without knowing the locations of topology boundary. For LT-PINNs, 32 sample points are allocated on each piece-wise line segment, and the boundary condition is enforced directly as a constraint term in the loss function. In contrast, for DT-PINNs, the boundary condition cannot be applied directly as it requires exact knowledge of boundary point locations. Instead, it is incorporated into the PDE loss term as specified in Eq. (5). A detailed summary of all training parameters for both DT-PINNs and LT-PINNs is provided in Table 6.

The reconstructed topologies and temperature fields are presented in Fig. 11. For both Setting A and Setting B, LT-PINNs yield more accurate topology reconstructions than DT-PINNs. Specifically, DT-PINNs generate irregular topologies that deviate significantly from the target primitive topologies, a shortfall primarily attributed to insufficient measurement data. In contrast, LT-PINNs successfully reconstruct temperature fields that closely align with the reference solutions. These results demonstrate the superior performance of LT-PINNs over DT-PINNs for topology optimization across different primitive geometric configurations. However, some deviations in the center location and geometric parameters of the reconstructed topology are observable in the LT-PINNs results, as quantified in Table 7. The worst-case performance occurs with the brick geometry, which exhibits a relative error between 22% and 22.67% for the length $L$ across both settings. This larger error is due to the brick's more complex shape and corresponding temperature field. Specifically, a high-temperature region exists at the brick's notch, whereas the temperature for the triangle and rectangle decays monotonically with increasing distance from the boundary.

Since DT-PINNs are unable to reconstruct the single primitive topology, we will not make a comparison on a combination of multiple primitive topologies. In fact, the key of multi-primitive topology optimization lies in managing the interaction and overlap between distinct topological patches. It is important to note that when primitives are non-overlapping and well-separated, the domain can be decomposed into isolated sub-domains, effectively reducing the problem to the single-primitive case which DT-PINNs already struggle to solve. Therefore, we instead focus our analysis on the advanced capabilities of LT-PINNs in handling the problem with multiple topology patches, specifically, patch interaction in regular arrays (Section 3.3) and patch overlap in irregular clusters (Section 3.4).

![15_265_218_1172_1802_0.jpg](images/15_265_218_1172_1802_0.jpg)

Fig. 11. Laplace's equation (Dirichlet boundary condition): comparison of DT-PINN and LT-PINN on different primitive geometric configurations: a) setting A; b) setting B.

Table 6

Laplace's equation (Dirichlet boundary condition): details of the PINN architectures and operating parameters.

<table><tr><td>Settings</td><td>DT-PINN</td><td>LT-PINN</td></tr><tr><td>#of layers</td><td>8</td><td>8</td></tr><tr><td>#of neurons per layer</td><td>256</td><td>256</td></tr><tr><td>activation function</td><td>Tanh</td><td>Tanh</td></tr><tr><td>optimizer</td><td>Adam</td><td>Adam</td></tr><tr><td>learning rate</td><td>$1 \times  {10}^{-4}$</td><td>$1 \times  {10}^{-4}$</td></tr><tr><td>training epoch</td><td>40,000 - 80,000</td><td>25,000 - 120,000</td></tr><tr><td>$N$</td><td>120×120</td><td>120 × 120</td></tr><tr><td>${N}_{d}$</td><td>100</td><td>100</td></tr><tr><td>${N}_{b}$</td><td>-</td><td>128</td></tr><tr><td>${N}_{t}$</td><td>-</td><td>-</td></tr><tr><td>${\lambda }_{p}$</td><td>1</td><td>0.1 - 1</td></tr><tr><td>${\lambda }_{d}$</td><td>$1 \times  {10}^{4}$</td><td>$1 \times  {10}^{4}$</td></tr><tr><td>${\lambda }_{b}$</td><td>-</td><td>$1 \times  {10}^{2}$</td></tr><tr><td>${\lambda }_{t}$</td><td>-</td><td>-</td></tr><tr><td>PDE</td><td>Eq. (19)</td><td>Eq. (19)</td></tr><tr><td>#of topology patches</td><td>-</td><td>1</td></tr><tr><td>#of GPUs</td><td>1</td><td>1</td></tr></table>

Table 7

The learned parameter settings of generated topologies via LT-PINNs.

<table><tr><td rowspan="2"></td><td colspan="3">Triangle</td><td colspan="3">Rectangle</td><td colspan="3">Brick</td></tr><tr><td>$\alpha$</td><td>${x}_{0}$</td><td>${y}_{0}$</td><td>$L$</td><td>${x}_{0}$</td><td>${y}_{0}$</td><td>$L$</td><td>${x}_{0}$</td><td>${y}_{0}$</td></tr><tr><td>Setting A</td><td>65.0°</td><td>0.01</td><td>0.02</td><td>0.55</td><td>-0.02</td><td>-0.01</td><td>0.61</td><td>0.06</td><td>0.09</td></tr><tr><td>Setting B</td><td>121.1°</td><td>0.05</td><td>0.02</td><td>1.46</td><td>-0.04</td><td>0.02</td><td>1.84</td><td>-0.08</td><td>0.01</td></tr></table>

### 3.3. Exploring LT-PINNs for complex topology optimization

The effectiveness of LT-PINNs have been thoroughly validated in previous sections. We now explore LT-PINN's potential in tackling more complex problems with multiple primitive topologies, further evaluating its topology optimization performance and robustness. In this section, to avoid convergence to local minima during topology optimization, we continue to employ circles of a fixed diameter as primitive topologies. This approach forms the basis for constructing more complex shapes from combinations of multiple topological patches. Using a fixed diameter reduces the number of geometric parameters, thereby facilitating the optimization process. Despite this parametric constraint, a cluster of circular patches can still generate a wide variety of shapes through properly configured interaction constraints and freedom to overlap. Furthermore, circles with a fixed diameter are easily converted into CAD models and simplify the manufacturing process.

The optimization of a complex topology represented by a multi-circle array, governed by highly nonlinear Navier-Stokes equations, is specifically carried out. The entire simulation domain is ${25D} \times  {15D}$ , similar to that in Fig. 4a, and the number of circles in the domain to be analyzed is set 2, 3, and 8, with the respective configurations shown in Fig. 12. For the 2-circle array, the center-to-center distance between the two circles is fixed at ${2.5D}$ . In the 3-circle array, each pair of adjacent circles maintains the same ${2.5D}$ spacing. For the 8-circle array, the circles are uniformly distributed on a circumference of the ${2.5D}$ -radius.

In Sections 3.1 and 3.2, only single circle and other primitive topology is considered in the test, thus it does not require topology loss function that is related to the constraints of multiple circles in Eq. (12). However, for multi-circle array topologies with self-similar patterns, we include the topology loss function in the LT-PINN besides the other loss functions, as topology loss function can reduce unnecessary optimization space and enhance LT-PINN's performance dealing with complex topology. In detail, for 2-circle array and 3-circle array, we consider the fixed distance constraint loss function $\left( {L}_{t}^{1}\right)$ , that forces distance between each pair of circles to be 2.5D, as shown in Fig. 12(a) and (b). For 8-circle array, the fixed distance constraint loss function $\left( {L}_{t}^{1}\right)$ forces distance between any circle to the center of 8-circle array to be ${2.5D}$ , as shown in Fig. 12(c). In addition to ${L}_{t}^{1}$ , we add the non-overlapping constraint loss function $\left( {L}_{t}^{2}\right)$ between each pair of circles for 8-circle array, which aims to maximize their pairwise distance so as to avoid overlapping.

The ROI for model training is adaptively adjusted to the dimensions of the multi-circle array, with a ${1.1D}$ extension in both width and height to ensure full coverage of all circles. The dimensions are described as: $\left\lbrack  {{L}_{x,\min } - {1.1D},{L}_{x,\max } + {1.1D}}\right\rbrack   \times \; \left\lbrack  {{L}_{y,\text{ min }} - {1.1D},{L}_{y,\text{ max }} + {1.1D}}\right\rbrack$ , where the minimal and maximum positions of the circle center in $x, y$ directions can be defined by ${L}_{x,\min },{L}_{x,\max },{L}_{y,\min },{L}_{y,\max }.$

![17_239_206_1228_533_0.jpg](images/17_239_206_1228_533_0.jpg)

Fig. 12. Time-dependent flow: configurations of multi-circle arrays: a) 2-circle array; b) 3-circle array; c) 8-circle array.

![17_332_889_1061_1182_0.jpg](images/17_332_889_1061_1182_0.jpg)

Fig. 13. Time-independent flow: loss history and $\gamma$ trajectory for 2-circle array: a-d) $\gamma$ trajectory at different training epochs; e) loss history.

#### 3.3.1. Time-independent flow problem

The flow problem governed by the time-independent Navier-Stokes equation is first investigated. The time-independent Navier-Stokes equation is given below:

$$
\nabla  \cdot  \mathbf{u} = 0
$$

$$
\left( {\mathbf{u} \cdot  \nabla }\right) \mathbf{u} =  - \nabla p + \frac{1}{Re}\Delta \mathbf{u} \tag{23}
$$

$$
{\left. u\right| }_{bc} = 0
$$

where $u$ is the velocity vector, $p$ is the pressure, and ${Re}$ is the Reynolds number. A no-slip boundary condition is imposed on all fixed circles.

Since it is a time-independent flow problem, we choose Reynolds number ${Re} = 1$ for laminar flow in Eq. (23). To generate training data, the edge conditions are: symmetry conditions on the top and bottom edges, zero pressure on the right edge, and uniform unit velocity on the left edge. Given the solution data from the numerical simulation, we extract sampled points for model training, with sparse measurement data being randomly selected from points outside the core part of ROI, i.e., $\left\lbrack  {{L}_{x,\min } - D,{L}_{x,\max } + D}\right\rbrack   \times  \left\lbrack  {{L}_{y,\min } - }\right. \; D,{L}_{y,{max}} + D\rbrack$ . Inside the core part of ROI, uniformly distributed PDE points are extracted for evaluating PDE loss in Eq. (11). Furthermore, to enforce the no-slip boundary conditions represented by Dirichlet boundary conditions on each circle, we uniformly sample 128 boundary points along four concentric rings within each circle at radii of 0.2, 0.3, 0.4, and 0.5 . The velocity at all sampled points is set to zero to satisfy the no-slip condition. Detailed description of the data generation and training settings for LT-PINNs are listed in Table A.1.

Initially, $\gamma$ (representing the circle centers) is randomly distributed within the domain. During training, the predicted topology of the multi-circle array, determined by $\gamma$ , gradually converges toward the reference configuration. The training loss history and $\gamma$ trajectory of the 2-circle array is shown in Fig. 13. As the training loss diminishes, $\gamma$ progressively aligns with the reference. When the loss becomes stable after around ${50}\mathrm{k}$ epochs, $\gamma$ also tends to be stable. Consistent findings on the loss history and $\gamma$ trajectory are also observed for the 3-circle array and the 8-circle array. Hence, we refrain from further elaboration. LT-PINNs enable simultaneous flow field prediction and topology optimization. Clearly, since $\gamma$ can be monitored concurrently with the loss history, we find that an early stopping strategy also becomes viable once $\gamma$ stabilizes, significantly reducing training time.

The final predicted velocity and pressure fields with the optimized topology via LT-PINNs are presented in Figs. 14-16. The results show excellent agreement between predicted and reference fields. For the 8-circle array configuration (Fig. 16(b)), the velocity field prediction successfully resolves the low-velocity zone in the central region, in close accordance with the reference solution. The pressure field prediction (Fig. 16(a)) similarly exhibits high fidelity, accurately reproducing the characteristic pressure gradient from higher left-side values to lower right-side pressures, precisely matching the reference distribution.

Across all test cases, LT-PINNs effectively reproduce the key fluid dynamic phenomena of flow diversion around multi-circle arrays. However, minor discrepancies persist between predicted and reference solutions. Notably, in the 3-circle array configuration (Fig. 15(a)), the predicted pressure field of the LT-PINN shows a high pressure anomaly in the central region. We attribute this deviation primarily to insufficient measurement data available during the training phase, which may have limited the model's ability to fully resolve the complex flow interactions. Although minor inaccuracies persist in the physical field predictions, the optimized topological configurations demonstrate excellent agreement with reference solutions in Fig. 17. This finding indicates that the LT-PINN ensures robust performance in complex topology optimization tasks.

Furthermore, the error of the predicted velocity and pressure fields is estimated by the relative ${L}_{2}$ error of ${128} \times  {128}$ uniformly distributed test sample points in the region (excluding those inside circles), i.e., $\left\lbrack  {{L}_{x,\min } - D,{L}_{x,\max } + D}\right\rbrack   \times  \left\lbrack  {{L}_{y,\min } - D,{L}_{y,\max } + D}\right\rbrack$ , as listed in Table 8.

As expected, the relative ${L}_{2}$ errors for multi-circle arrays are higher than those observed for the single-circle case (Section 3.1), where the governing PDEs are significantly simpler. However, since the relative ${L}_{2}$ error can be disproportionately amplified when the field's absolute mean value is small, we additionally employ the normalized mean absolute error (NMAE) [47] as a more robust error metric. The NMAE mitigates bias induced by low-magnitude reference fields and provides a clearer assessment of prediction accuracy.

The NMAE results, presented in Table 9, illustrate that the maximum error occurs in the pressure prediction for the 3-circle array (NMAE = 0.1025), corroborating the earlier observation of elevated pressure deviations in Fig. 15(a). Nevertheless, the remaining NMAE values remain below 0.1, indicating that the predicted velocity and pressure fields generally deviate by less than 10% relative to the reference solution. For practical engineering applications, this level of agreement illustrates sufficient accuracy, confirming the reliability of LT-PINNs for time-independent flow field prediction in complex topological configurations.

#### 3.3.2. Time-dependent flow problem

We then consider a even more non-linear case, time-dependent Navier-Stokes for flow problem, characterized as:

$$
\nabla  \cdot  \mathbf{u} = 0
$$

$$
{\mathbf{u}}_{t} + \left( {\mathbf{u} \cdot  \nabla }\right) \mathbf{u} =  - \nabla p + \frac{1}{\operatorname{Re}}\Delta \mathbf{u}, \tag{24}
$$

$$
{\left. u\right| }_{bc} = 0
$$

where ${\mathbf{u}}_{t}$ is the gradient of velocity to time.

![19_395_158_926_1016_0.jpg](images/19_395_158_926_1016_0.jpg)

Fig. 14. Time-independent flow: predicted velocity and pressure field via LT-PINN for 2-circle array: a) pressure; b) $x$ -velocity; c) $y$ -velocity.

Table 8

Time-independent flow: relative ${L}_{2}$ error of the predicted velocity and pressure via LT-PINN w/ topology loss.

<table><tr><td>Relative ${L}_{2}$ error</td><td>2-circle</td><td>3-circle</td><td>8-circle</td></tr><tr><td>Pressure</td><td>0.3995</td><td>0.9656</td><td>0.4253</td></tr><tr><td>$x$ -velocity</td><td>0.2282</td><td>0.1620</td><td>0.1919</td></tr><tr><td>$y$ -velocity</td><td>0.3151</td><td>0.3421</td><td>0.2849</td></tr></table>

Table 9

Time-independent flow: normalized mean absolute error (NMAE) of predicted velocity and pressure via LT-PINN w/ topology loss.

<table><tr><td>NMAE</td><td>2-circle</td><td>3-circle</td><td>8-circle</td></tr><tr><td>Pressure</td><td>0.0387</td><td>0.1025</td><td>0.0559</td></tr><tr><td>$x$ -velocity</td><td>0.0722</td><td>0.0533</td><td>0.0510</td></tr><tr><td>$y$ -velocity</td><td>0.0391</td><td>0.0634</td><td>0.0335</td></tr></table>

Since the circles are stationary, there is no need to infer the circle boundary change over time. Therefore, we omit the time terms in the Navier-Stokes equation by converting it into the pressure Poisson equation:

$$
{\Delta p} =  - \nabla  \cdot  \left( {\mathbf{u} \otimes  \mathbf{u}}\right) ,
$$

$$
{\left. \mathbf{u}\right| }_{bc} = 0 \tag{25}
$$

where $\otimes$ is the outer product.

![20_384_155_938_1322_0.jpg](images/20_384_155_938_1322_0.jpg)

Fig. 15. Time-independent flow: predicted velocity and pressure field via LT-PINN for 3-circle array: a) pressure; b) $x$ -velocity; c) $y$ -velocity.

For time-dependent flow problems, we employ ${Re} = {100}$ in Eq. (24) to capture unsteady vortex shedding behind circular structures [48], while maintaining the same ROI edge conditions as the steady-state case described in Section 3.3.1 for data generation. The LT-PINN formulation employs the instantaneous pressure Poisson equation (Eq. (25)) in its PDE loss function (Eq. (11)), eliminating the need for time-resolved training data. Spatial training data at a time instance are sampled using the same methodology as the time-independent case (Section 3.3.1), with identical initialization procedures for the learnable parameters $\gamma$ . To maintain focus, we evaluate only the most challenging 8-circle array configuration for time-dependent analysis. Complete details of data preparation and PINN training configurations are provided in Table A.1.

To better understand the training process, the training loss history and $\gamma$ trajectory of 8-circle array is shown in Fig. 18. Upon reaching approximately ${50}\mathrm{k}$ epochs, both the loss and predicted topology achieve stability. It is worth mentioning that the final predicted topology exhibits strong agreement with the reference configuration, validating the desired accuracy of the LT-PINN to infer topology in time-dependent flow problem.

The predicted velocity and pressure fields for the time-dependent flow case are presented in Fig. 19. Comparative analysis reveals generally good agreement between the LT-PINN predictions and the reference solution. Specifically, the pressure field prediction successfully reproduces the characteristic high-pressure region upstream of the multi-circle array and the corresponding low-pressure wake region, demonstrating consistency with the reference data. Regarding velocity field predictions, the LT-PINN accurately captures the formation of two high-velocity jets through the array, as evidenced in Fig. 19(b). However, certain flow features exhibit discrepancies. Most notably, while the reference solution shows curved jet trajectories with significant vertical velocity components, the LT-PINN prediction yields straighter jet paths with more uniform vertical velocity distributions (Fig. 19(c)). The observed differences in flow field predictions, particularly in the vertical velocity components, indicate potential areas for model improvement. We hypothesize that the inclusion of additional measurement data within the domain would enhance the accuracy of flow pattern predictions.

![21_398_173_924_1424_0.jpg](images/21_398_173_924_1424_0.jpg)

Fig. 16. Time-independent flow: predicted velocity and pressure field via LT-PINN for 8-circle array: a) pressure; b) x-velocity; c) y-velocity.

![21_269_1712_1177_381_0.jpg](images/21_269_1712_1177_381_0.jpg)

Fig. 17. Time-independent flow: predicted topology via LT-PINNs w/ topology loss: a) 2-circle array; b) 3-circle array; c) 8-circle array.

![22_368_161_994_1208_0.jpg](images/22_368_161_994_1208_0.jpg)

Fig. 18. Time-dependent flow: loss history and $\gamma$ trajectory for 8-circle array: a-d) $\gamma$ trajectory at different training epochs; e) loss history.

Table 10

Time-dependent flow: predicted lift, drag, and lift-drag ratio via LT-PINN.

<table><tr><td>Force on multi-circle array</td><td>Reference</td><td>LT-PINN</td></tr><tr><td>Lift</td><td>0.0578</td><td>0.0574</td></tr><tr><td>Drag</td><td>3.7337</td><td>1.3827</td></tr><tr><td>Lift-drag ratio</td><td>0.0155</td><td>0.0415</td></tr></table>

The hydrodynamic forces on the circles provide critical quantitative measures of flow behavior. We assess predictive performance by computing pressure-induced lift and drag forces for the 8-circle array (Table 10). The results show good agreement in lift force predictions, but a substantial 62.97% underestimation of drag forces. As a result, this discrepancy produces a corresponding 1.677- fold overestimation of the lift-to-drag ratio relative to the reference solution.

Given the observed discrepancies in both flow patterns and force predictions for the 8-circle array configuration, we employ the NMAE as a metric to systematically evaluate the predictive accuracy of the LT-PINN. The NMAE is calculated on the test sample points extracted from the ROI as described in Section 3.3.1, and results are listed in Table 11. The analysis indicates that while the $x$ -velocity component exhibits a relatively elevated NMAE of 0.2032 due to unresolved fine-scale flow features between adjacent circles, both pressure and $y$ -velocity predictions achieve satisfactory accuracy (errors <10.2%) within standard engineering tolerances. These results demonstrate LT-PINN's capability for concurrent topology optimization and PDE solution prediction in time-dependent flows.

![23_388_156_933_1329_0.jpg](images/23_388_156_933_1329_0.jpg)

Fig. 19. Time-dependent flow: predicted velocity and pressure field via LT-PINN for 8-circle array: a) pressure; b) $x$ -velocity; c) $y$ -velocity.

Table 11

Time-dependent flow: NMAE of the

predicted velocity and pressure via

LT-PINN w/ topology loss.

<table><tr><td>NMAE</td><td>Time-dependent flow</td></tr><tr><td>Pressure</td><td>0.1011</td></tr><tr><td>$x$ -velocity</td><td>0.2032</td></tr><tr><td>$y$ -velocity</td><td>0.0595</td></tr></table>

### 3.4. Application on flow velocity rearrangement

The upstream flow velocity profile represents a critical characteristic influencing numerous flow phenomena, including turbulence development [49], flow measurement accuracy [50], biofilm growth in water distribution systems [51], and gas cyclone separator performance [52]. Consequently, controlled manipulation of upstream velocity profiles to achieve target downstream velocity profiles demonstrates substantial engineering potential.

![24_278_159_1159_920_0.jpg](images/24_278_159_1159_920_0.jpg)

Fig. 20. Flow rearrangement: generated topology and rearranged downstream velocity profile via LT-PINN: a) initial topology; b) generated topology; c) rearranged downstream velocity profile.

In this section, we formulate a flow velocity rearrangement task for illustrating the practical feasibility of the LT-PINN, where the LT-PINN can generate an optimal topology based solely on the upstream velocity profile without prior topological knowledge, to achieve a specified target downstream velocity profile. In detail, the ROI is expanded to be ${25D} \times  {15D}$ . Boundary conditions consist of a uniform inlet velocity profile at the left edge $\left( {x\text{ -velocity } = 1, y\text{ -velocity } = 0}\right)$ , periodic conditions at the top and bottom edges, and a target downstream $x$ -velocity profile $u = \sin \left( {{2\pi x}/{15}}\right)  + 1$ at the right edge, with Reynolds number ${Re} = 1$ . Data loss function are implemented to enforce both the inlet/outlet velocity profiles and the periodicity condition between the top and bottom edges, while no pressure-based data loss is applied. For topology optimization, LT-PINN utilizes 48 circles with allowed overlap, providing sufficient design freedom without topology loss constraints in this configuration. Since the ROI in this case is much larger than that of other cases, resulting in a larger amount of sampled data, we employ 4xGPUs as parallel computing resource for this case. Complete details regarding data sampling and PINN training parameters are provided in Table A.1.

Fig. 20 shows the optimized topology and resulting downstream velocity profile from LT-PINN prediction. Initially, 48 randomly distributed circles populate the domain (Fig. 20(a)). During optimization, these patches undergo substantial spatial reorganization, ultimately coalescing into an intricate cluster in the bottom-right region (Fig. 20(b)). The cluster's exact configuration can be precisely determined, since each circle's position is explicitly defined. Furthermore, as demonstrated in Fig. 20(c), the downstream x-velocity profile successfully achieves the target sinusoidal distribution.

The conversion of the detailed $x$ -velocity profile from a uniform upstream velocity to a target sine-shaped downstream velocity is depicted in Fig. 21. It is evident that the $x$ -velocity profile conversion occurs more rapidly in the lower $y$ -region, completing by $x = 8$ , while the upper region requires a longer development length. This asymmetric conversion pattern results from the topology’s cluster configuration in the bottom-right corner (centered near $x = 5$ ).

Based on the cluster topology generated by the LT-PINN, a Computer-Aided Design (CAD) model is constructed to facilitate Computational Fluid Dynamics (CFD) simulations for the purpose of demonstrating the compatibility of the LT-PINN with CAD and CFD, in addition to generate reference validation data. Equivalent velocity boundary conditions are applied to the left, top, and bottom edges with a corresponding pressure boundary condition on the right edge. Under the prescribed boundary conditions, a comprehensive comparison between the predicted flow fields and reference solutions is presented in Fig. 22. As observed, LT-PINN successfully captures the low $x$ -velocity region downstream of the cluster, demonstrating good agreement with the reference data. However, notable discrepancies are observed in several aspects: (1) The $x$ -velocity profile near the right edge exhibiting a more pronounced gradient between the high-velocity (top-right) and low-velocity (bottom-right) regions in LT-PINN predictions compared to the reference solution (Fig. 22(b)); and (2) Evident differences in both pressure and $y$ -velocity fields, particularly upstream of the cluster where the reference solution shows elevated pressure and $y$ -velocity magnitudes relative to the LT-PINN predictions.

![25_467_187_772_498_0.jpg](images/25_467_187_772_498_0.jpg)

Fig. 21. Flow rearrangement: predicted velocity at different $x$ -positions via LT-PINN.

![25_295_793_1121_1303_0.jpg](images/25_295_793_1121_1303_0.jpg)

Fig. 22. Flow rearrangement: predicted velocity and pressure field via LT-PINN: a) pressure; b) $x$ -velocity; c) $y$ -velocity.

![26_266_170_1176_1343_0.jpg](images/26_266_170_1176_1343_0.jpg)

Fig. 23. Flow rearrangement: comparison between the LT-PINN and reference on the undefined edge conditions: a, b, c) are undefined pressure edge conditions; d, e) are undefined velocity edge conditions.

The above discrepancies are hypothesized to originate from differences in imposing four edges' conditions between the LT-PINN and the reference solution. As illustrated in Fig. 23, the undefined edge conditions, comprising velocity specifications at the right edge and pressure conditions along the remaining three edges, exhibit notable variations. Quantitative analysis reveals substantial differences in the undefined pressure edge conditions, while the velocity edge conditions at the right edge demonstrate relatively minor deviations.

Specifically, the reference solution's downstream $x$ -velocity profile displays a sinusoidal pattern comparable to the LT-PINN results. Furthermore, the $y$ -velocity components at $x =  - {7.5}$ and $x = {7.5}$ show close agreement between both sets of results. These observations indicate that pressure edge condition differences constitute the primary source of flow field discrepancies between the reference solution and LT-PINN results. However, from the perspective of topological generation capability, particularly in converting uniform upstream velocity into the target sine-shape downstream velocity profile, LT-PINN predictions match with the reference solution, as illustrated in Fig. 23d.

#### 3.5.Key insights

This study presents a novel LT-PINN that simultaneously performs topology optimization and solves governing PDEs. The major novelty of the LT-PINN lies in their use of primitive topology patches, a concept similar to those in the Moving Morphable Component (MMC) [33] and Moving Morphable Bar (MMB) [53]. The proposed approach provides precise topology boundaries, thereby avoiding the numerical instabilities of boundary reconstruction inherent in implicit topology representation methods, such as the wavy boundary problem in FF-PINNTO [26] and the irregular boundary deviations observed in previous study [27] and our case studies using DT-PINN. By integrating the meshless advantages of PINNs with Lagrangian topology optimization techniques, LT-PINN offers an efficient, effective, and unified approach for boundary-focused engineering optimization problems.

We first benchmark LT-PINNs against state-of-the-art DT-PINNs using two canonical test cases: (1) 2D elastic equations with Dirichlet boundary conditions and (2) 2D Laplace's equations with Neumann boundary conditions. The comparative analysis demonstrates LT-PINNs' superior accuracy, achieving a 66.53% reduction in displacement prediction error (relative ${L}_{2}$ error) for elastic equations and a remarkable 99.42% reduction relative ${L}_{2}$ error for predicted heat flux across the boundary for Laplace’s equations compared to DT-PINNs. These results conclusively establish LT-PINNs' enhanced capability for PDE solution accuracy across different boundary condition types. Furthermore, LT-PINNs demonstrate superior topology inference accuracy compared to DT-PINNs, particularly for Laplace's equations with Neumann boundary conditions. While DT-PINNs produce irregular topological configurations due to their inability to properly incorporate Neumann conditions in their total loss function (Eq. (5)), LT-PINNs achieve precise topology reconstruction through correct boundary condition encoding (Eq. (11)). Another key advantage of LT-PINNs is their automatic generation of well-defined topological boundaries, in contrast to DT-PINNs which require subjective manual interpolation of density thresholds. This automated approach eliminates a significant source of human-induced variability, ensuring more reliable and reproducible topology inference.

Having validated LT-PINNs' effectiveness across different boundary conditions through comparison with DT-PINNs on fundamental test cases of single circle geometry, we extend the performance investigation on various primitive topologies, including triangles with varying angles, rectangles and bricks with varying aspect ratios, for 2D Laplace's equations with Dirichlet boundary conditions, by utilizing less than 1% sparse measurement data. By leveraging piece-wise lines to describe primitive topologies, LT-PINNs are trained to identify the associated unknown geometric parameters, thereby simultaneously reconstructing both the topology and the temperature field. A comparison with DT-PINNs demonstrates that LT-PINNs achieve more accurate reconstructions of the unknown topology and temperature field. Notably, DT-PINNs exhibit severe irregular boundary deviations across all tested primitive topologies, a shortfall attributable to insufficient measurement data. These results confirm the performance robustness of LT-PINNs for reconstructing various primitive topologies.

Furthermore, we explore the capability of LT-PINNs on more challenging applications: complex topology optimization problems and highly nonlinear Navier-Stokes equations for 2D flow around multi-circle arrays. Notably, LT-PINNs operate using only edge data from the four edges of ROI, without any interior measurements, presenting a significant computational challenge. For time-independent flow at ${Re} = 1$ , LT-PINNs accurately reconstruct 2-,3-, and 8-circle array configurations, with velocity and pressure field predictions maintaining the NMAE below ${11}\%$ of each physical field’s absolute range. In time-dependent flow $\left( {{Re} = {100}}\right)$ past an 8-circle array, while topological inference remains accurate, velocity and pressure predictions show higher NMAE up to 20.32%. However, LT-PINNs achieve satisfactory accuracy in predicting lift forces in the 8-circle array case. The future work may incorporate additional measurement data for LT-PINNs training to further improve prediction fidelity.

Validation across 2- to 8-circle array configurations indeed confirms LT-PINNs' capability for complex topology optimization. However, those complex topology contains self-similar patterns, the characteristics of which are inherently encoded in the topology loss function. To further assess the generalizability of LT-PINNs in scenarios without such prior knowledge, we conduct an additional study focusing on a flow velocity rearrangement problem, a task of significant relevance to various downstream fluid dynamics applications. In this experiment, 48 randomly distributed circle patches are initialized within the computational domain for topology optimization. The objective is to evolve these patches into an optimal configuration capable of converting a uniform upstream velocity profile into a specified sinusoidal downstream velocity profile. Notably, the topology loss function is intentionally excluded from the optimization process, and the circle patches are allowed to overlap. As training progresses, the circle patches exhibit a tendency to coalesce, ultimately forming an irregular cluster topology near the downstream right edge of the domain. Despite the absence of explicit topological constraints, the resultant downstream velocity profile demonstrates strong agreement with the target sinusoidal distribution. While geometrically irregular, the explicit parameterization of each patch enables straightforward CAD reconstruction, demonstrating LT-PINNs' strong potential for engineering design integration. Complementary computational fluid dynamics (CFD) simulations are also performed to demonstrating the compatibility of LT-PINNs with CAD and CFD, and validate physical realizability of the inferred topology. The resulting downstream velocity profiles exhibits close alignment with the target sinusoidal distribution, further validating the efficacy of the optimized cluster topology in this flow velocity rearrangement task.

Throughout our numerical experiments presented in this study, we have limited our investigation to primitive topological patches. Future work should explore extensions to arbitrary patch geometries with deformable boundaries described by higher-order parametric curves, for example B-spline curve, Bézier and NURBS curves. Besides, LT-PINN applications should be expanded to more realistic three-dimensional problems, such as metamaterial design [54], offshore structure optimization [55], sediment transport management [56], and artificial reef optimization [57].

## 4. Conclusion

To address the critical need for simultaneous topology optimization and PDE solution, we present LT-PINNs, a novel and unified framework that combines topological design and optimization with PDE solving in a meshless Lagrangian formulation. Our approach demonstrates substantial improvements in both accuracy and versatility, surpassing current state-of-the-art counterpart, i.e., DT-PINNs. Its feasibility and engineering potential are further demonstrated through a series of experiments. The key innovations of this work include:

1. LT-PINNs eliminate the error-prone manual boundary interpolation required in DT-PINNs, leading to a clearer and more efficient topology inference;

2. LT-PINNs precisely encode arbitrary boundary conditions through boundary condition loss function, resulting in a substantial reduction of the relative ${L}_{2}$ error in the predicted PDE solution;

3. For complex topology systems, LT-PINNs can incorporate topology loss function, particularly effective for self-similarity topology patterns;

4. For flow velocity rearrangement tasks, LT-PINNs can generate manufacturable irregular topologies directly compatible with CAD, without the measurement data or prior knowledge of topological features.

Future work will focus on extending LT-PINNs to cope with more general topologies with deformable boundaries and 3-dimensional problems, thereby significantly broadening their applicability.

## CRediT authorship contribution statement

Yuanye Zhou: Writing - review & editing, Writing - original draft, Visualization, Validation, Methodology, Investigation, Formal analysis, Data curation, Conceptualization; Zhaokun Wang: Writing - review & editing, Data curation; Kai Zhou: Writing - review & editing, Writing - original draft, Visualization, Supervision, Software, Resources, Project administration, Funding acquisition, Formal analysis, Conceptualization; Hui Tang: Supervision; Xiaofan Li: Writing - review & editing.

## Data availability

I have shared my data and code at: https://github.com/cloud2009/LT-PINN

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgments

The research is financially supported by the start-up grant from The Hong Kong Polytechnic University and the research project funding from the Research Institute for Sustainable Urban Development (RISUD) at The Hong Kong Polytechnic University. In addition, HT would like to acknowledge financial support from Research Grants Council of Hong Kong under General Research Fund (15218421). Finally, we are grateful to Prof. Xin Bian and Mr. Yongzheng Zhu from Zhejiang Univeristy for their insightful discussions on PINNs.

## Appendix A. Computational setup

In this section, the computational setup, including the architectures of neural networks, optimizer, etc., for all test cases, is presented for interested readers to replicate analysis. Details are summarized in the Table A. 1 in addition to the case descriptions in Section 3.1.

Table A. 1

Details for the LT-PINNs in different test cases.

<table><tr><td>Settings</td><td>Section 3.3.1 2-circle</td><td>Section 3.3.1 3-circle</td><td>Section 3.3.1 8-circle</td><td>Section 3.3.2 8-circle</td><td>Section 3.4 48-circle</td></tr><tr><td>#of layers</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td></tr><tr><td>#of neurons per layer</td><td>64</td><td>64</td><td>64</td><td>64</td><td>64</td></tr><tr><td>activation function</td><td>Tanh</td><td>Tanh</td><td>Tanh</td><td>Tanh</td><td>Tanh</td></tr><tr><td>optimizer</td><td>Adam</td><td>Adam</td><td>Adam</td><td>Adam</td><td>Adam</td></tr><tr><td>learning rate</td><td>0.0001</td><td>0.0001</td><td>0.0001</td><td>0.0001</td><td>0.0001</td></tr><tr><td>training epoch</td><td>100,000</td><td>100,000</td><td>100,000</td><td>100,000</td><td>100,000</td></tr><tr><td>$N$</td><td>270×120</td><td>249×270</td><td>${420} \times  {420}$</td><td>${420} \times  {420}$</td><td>${1500} \times  {900}$</td></tr><tr><td>${N}_{b}$</td><td>1,024</td><td>1,536</td><td>4,096</td><td>4096</td><td>20,480</td></tr><tr><td>${N}_{d}$</td><td>3,389</td><td>3,682</td><td>5,373</td><td>5373</td><td>800</td></tr><tr><td>${N}_{t}$</td><td>2</td><td>3</td><td>8</td><td>8</td><td>48</td></tr><tr><td>${\lambda }_{p}$</td><td>$2 \times  {10}^{3}$</td><td>$2 \times  {10}^{3}$</td><td>$2 \times  {10}^{3}$</td><td>$2 \times  {10}^{3}$</td><td>$2 \times  {10}^{3}$</td></tr><tr><td>${\lambda }_{b}$</td><td>${10}^{4}$</td><td>${10}^{4}$</td><td>${10}^{4}$</td><td>${10}^{4}$</td><td>${10}^{4}$</td></tr><tr><td>${\lambda }_{d}$</td><td>${10}^{4}$</td><td>${10}^{4}$</td><td>${10}^{4}$</td><td>${10}^{4}$</td><td>${10}^{4}$</td></tr><tr><td>${\lambda }_{t}$</td><td>${10}^{4}$</td><td>${10}^{4}$</td><td>${10}^{2}$</td><td>${10}^{2}$</td><td>-</td></tr><tr><td>PDE</td><td>Eq. (23)</td><td>Eq. (23)</td><td>Eq. (23)</td><td>Eq. (25)</td><td>Eq. (23)</td></tr><tr><td>#of topology patches</td><td>2</td><td>3</td><td>8</td><td>8</td><td>48</td></tr><tr><td>#of GPUs</td><td>1</td><td>1</td><td>1</td><td>1</td><td>4</td></tr></table>

## Appendix B. Comparison of different topology representation approaches

Two simple cases, 1-dimensional (1D) step geometry and 2-dimensional (2D) circle geometry, are chosen for evaluating and comparing different topology representation approaches, with particular emphasis on the reconstruction accuracy at the topology boundaries. The considered representation approaches include the density approach (used in DT-PINN with and without transformation function), Lagrangian approach (used in LT-PINN), and Fourier approach (used in FF-PINNTO [26]).

The geometric function of 1D step geometry is defined as:

$$
\phi \left( x\right)  = \left\{  \begin{array}{ll} 0, & \left( {x - {x}_{0}}\right)  < 0 \\  {0.5}, & \left( {x - {x}_{0}}\right)  = 0 \\  1, & \left( {x - {x}_{0}}\right)  > 0, \end{array}\right. \tag{B.1}
$$

where ${x}_{0}$ is the boundary of step geometry and $x \in  \left\lbrack  {0,1}\right\rbrack$ .

The geometric function of 2D circle geometry is defined as:

$$
\phi \left( {x, y}\right)  = \left\{  \begin{array}{ll} 0, & \sqrt{{\left( x - {x}_{0}\right) }^{2} + {\left( y - {y}_{0}\right) }^{2}} < {r}_{0} \\  {0.5}, & \sqrt{{\left( x - {x}_{0}\right) }^{2} + {\left( y - {y}_{0}\right) }^{2}} = {r}_{0} \\  1, & \sqrt{{\left( x - {x}_{0}\right) }^{2} + {\left( y - {y}_{0}\right) }^{2}} > {r}_{0}, \end{array}\right. \tag{B.2}
$$

where ${x}_{0},{y}_{0}$ is the center of circle, ${r}_{0}$ is the radius of circle, $x \in  \left\lbrack  {0,1}\right\rbrack$ , and $y \in  \left\lbrack  {0,1}\right\rbrack$ .

For 1D step geometry, ${x}_{0} = {0.5}$ ; for 2D circle geometry, ${x}_{0} = {y}_{0} = {0.5}$ and ${r}_{0} = {0.25}$ . Therefore, the boundary curve function for 1D step geometry is ${F}_{\gamma }\left( x\right)  = x = {0.5}$ and the boundary curve function for 2D circle geometry is ${F}_{\gamma }\left( {x, y}\right)  = \sqrt{{\left( x - {0.5}\right) }^{2} + {\left( y - {0.5}\right) }^{2}} = {0.25}$ .

For density approach without transformation function, the geometric function $\left( \phi \right)$ is approximated by the density $\left( \rho \right)$ as the output of neural network. For density approach with transformation function, a transformation function is added, defined as:

$$
\widehat{\rho }\left( \mathbf{x}\right)  = \frac{1}{1 + {e}^{-{c\rho }\left( \mathbf{x}\right) }} \tag{B.3}
$$

where $\rho \left( \mathbf{x}\right)$ is the output of neural network without transformation function, $\widehat{\rho }\left( \mathbf{x}\right)$ is the transformed density, $c$ is a fixed coefficient of transformation function, and $\mathbf{x}$ is the vector of neural network inputs.

The density approach is an indirect approach to learn the unknown geometry as it relies on approximating the geometric function ( $\phi$ ) defined in the whole domain. While for Lagrangian approach, it is a direct approach. The boundary curve function $\left( {F}_{\gamma }\right)$ that explicitly defines the geometry is approximated by the learnable geometric parameters $\left( \gamma \right)$ in neural network.

For Fourier approach, the input vector $\mathbf{x}$ is mapped to a Fourier space $\left( \widehat{x}\right)$ with $M$ random deviations as described in FF-PINNTO [26] and the geometric function $\left( \phi \right)$ is also approximated by the density as the output of neural network.

Therefore, the corresponding loss functions for these methods are:

$$
{L}_{d}^{\rho } = \frac{1}{{N}_{d}}\mathop{\sum }\limits_{{i = 1}}^{{N}_{d}}{\left( \rho \left( \mathbf{x}\right)  - {\phi }^{ * }\left( \mathbf{x}\right) \right) }_{i}^{2},\text{ Density }
$$

$$
{L}_{d}^{L} = {\left( \gamma  - {\gamma }^{ * }\right) }^{2},\;\text{ Lagrangian } \tag{B.4}
$$

$$
{L}_{d}^{F} = \frac{1}{{N}_{d}}\mathop{\sum }\limits_{{i = 1}}^{{N}_{d}}{\left( \rho \left( \widehat{\mathbf{x}}\right)  - {\phi }^{ * }\left( \mathbf{x}\right) \right) }_{i}^{2}\text{ , Fourier }
$$

where ${L}_{d}^{\rho },{L}_{d}^{L}$ , and ${L}_{d}^{F}$ are the data loss functions of density approach, Lagrangian approach and Fourier approach, respectively. ${\phi }^{ * }\left( x\right)$ is the known reference geometric function, $\widehat{x}$ is location of the data in Fourier space, and ${\gamma }^{ * }$ is the known reference geometric parameter, such as the ${x}_{0},{y}_{0},{r}_{0}$ in Eq. (B.2). ${N}_{d}$ is the number of data points. When transformation function is added, $\rho$ is replaced by $\widehat{\rho }$ as defined in Eq. (B.3).

The number of neurons in all neural networks used in these approaches is 64 and the number of hidden layers is 4 . The activation functions are all tanh function, except the Fourier approach, which uses sine activation function. $M$ is set to be 7 for 1D step geometry and 13 for $2\mathrm{D}$ circle geometry. The optimizer is Adam and learning rate is $1 \times  {10}^{-4}$ for $1\mathrm{D}$ step geometry and $1 \times  {10}^{-5}$ for $2\mathrm{D}$ circle geometry. $c$ is 10 in the transformation function.

A set of coarse discrete data points with uniform grid size of 100 is generated according to the defined geometric functions for training. The training is running 100,000 epochs for 1D step geometry and 200,000 for 2D circle geometry. After training is complete, the neural networks are tested on a high-resolution grid with a grid size of 1,000 . The Lagrangian approach can learn the boundary curve function directly. To map this back to the geometric function, we use the geometric distance function $\delta$ (Eq. (6)) with a sufficiently large sharpness parameter $\beta$ .

A zoomed-in view of the approximated 1D step geometric function is shown in Fig. B.1. clearly, the Lagrangian approach provides the best fit to the reference results among all approaches. The density approach with a transformation function performs better than the approach without one and the Fourier approach, as the transformation sharpens the predicted geometric function at the boundary $\left( {x = {0.5}}\right)$ .

Fig. B. 2 shows the absolute error (AE) between the predicted and reference geometric functions for the 2D circle case. The high AE for both the Lagrangian and the transformed density approaches is concentrated near the circle boundary. However, the high-error region is slightly wider for the transformed density approach than for the Lagrangian method. In contrast, the density approach without transformation and the Fourier approach exhibit error throughout the entire domain. Notably, high error regions also appear near the boundary for these methods, a phenomenon likely caused by Gibbs oscillation [58] due to the sharp discontinuity at the boundary. These results confirm that the Lagrangian approach most accurately fits the reference function and the transformation function significantly improves the performance of the density approach.

![30_497_166_712_488_0.jpg](images/30_497_166_712_488_0.jpg)

Fig. B.1. Comparison between different topology representation approaches on 1D step geometry.

![30_372_731_959_702_0.jpg](images/30_372_731_959_702_0.jpg)

Fig. B.2. Comparison between different topology representation approaches on 2D circle geometry (absolute error to reference).

Table B.1

Relative ${L}_{2}$ error of different topology representation approaches.

<table><tr><td></td><td>Density</td><td>Density*</td><td>Lagrangian</td><td>Fourier</td></tr><tr><td>1D Step</td><td>${4.037} \times  {10}^{-2}$</td><td>${3.294} \times  {10}^{-2}$</td><td>${4.194} \times  {10}^{-4}$</td><td>${4.140} \times  {10}^{-2}$</td></tr><tr><td>2D Circle</td><td>${1.759} \times  {10}^{-2}$</td><td>${1.692} \times  {10}^{-2}$</td><td>${4.644} \times  {10}^{-3}$</td><td>${1.924} \times  {10}^{-2}$</td></tr></table>

The relative ${L}_{2}$ errors of the predicted geometric functions for two cases are listed in Table B.1. As can be seen, the density approach with transformation performs slightly better than the density approach without transformation and the Fourier approach. The Lagrangian approach has the lowest relative ${L}_{2}$ error compared to the other approaches for both cases. Specifically, the Lagrangian approach achieves at least an order of magnitude lower error, demonstrating its superior accuracy for topology representation.

## References

[1] M.P. Bendsøe, Optimal shape design as a material distribution problem, Struct. Optim. 1 (1989) 193-202.

[2] C.B.W. Pedersen, P. Allinger, Industrial implementation and applications of topology optimization and future needs, in: IUTAM Symposium on Topological Design Optimization of Structures, Machines and Materials: Status and Perspectives, Springer, 2006, pp. 229-238.

[3] J.-H. Zhu, W.-H. Zhang, L. Xia, Topology optimization in aircraft and aerospace structures design, Arch. Comput. Methods Eng. 23 (2016) 595-622.

[4] R.J. Yang, A.I. Chahande, Automotive applications of topology optimization, Struct. Optim. 9 (1995) 245-249.

[5] X. Huang, Y.M. Xie, G. Lu, Topology optimization of energy-absorbing structures, Int. J. Crashworthiness 12 (6) (2007) 663-675.

[6] N. Wu, S. Li, B. Zhang, C. Wang, B. Chen, Q. Han, J. Wang, The advances of topology optimization techniques in orthopedic implants: a review, Med. Biol. Eng. Comput. 59 (9) (2021) 1673-1689.

[7] S. Mukherjee, D. Lu, B. Raghavan, P. Breitkopf, S. Dutta, M. Xiao, W. Zhang, Accelerating large-scale topology optimization: state-of-the-art and challenges, Arch. Comput. Methods Eng. 28 (2021) 1-23.

[8] T. Borrvall, J. Petersson, Large-scale topology optimization in 3D using parallel computing, Comput. Methods Appl. Mech. Eng. 190 (46-47) (2001) 6201-6229.

[9] J. Paris, I. Colominas, F. Navarrina, M. Casteleiro, Parallel computing in topology optimization of structures with stress constraints, Comput. Struct. 125 (2013) 62-73.

[10] O. Amir, M. Stolpe, O. Sigmund, Efficient use of iterative solvers in nested topology optimization, Struct. Multidiscip. Optim. 42 (2010) 55-72.

[11] Y.Y. Kim, G.H. Yoon, Multi-resolution multi-scale topology optimization-a new paradigm, Int. J. Solids Struct. 37 (39) (2000) 5529-5559.

[12] J. Park, A. Sutradhar, A multi-resolution method for 3D multi-material topology optimization, Comput. Methods Appl. Mech. Eng. 285 (2015) 571-586.

[13] S. Shin, D. Shin, N. Kang, Topology optimization via machine learning and deep learning: a review, J. Comput. Des. Eng. 10 (4) (2023) 1736-1766.

[14] C. Kim, J. Lee, J. Yoo, Machine learning-combined topology optimization for functionary graded composite structure design, Comput. Methods Appl. Mech. Eng. 387 (2021) 114-158.

[15] Y. Zhang, C. Jia, X. Liu, J. Xu, B. Guo, Y. Wang, S. Zhang, Enhancing topology optimization with adaptive deep learning, Comput. Struct. 305 (2024) 107-527.

[16] F.V. Senhora, H. Chi, Y. Zhang, L. Mirabella, T.L.E. Tang, G.H. Paulino, Machine learning for topology optimization: physics-based learning through an independent training strategy, Comput. Methods Appl. Mech. Eng. 398 (2022) 115-116.

[17] J. Yin, Z. Wen, S. Li, Y. Zhang, H. Wang, Dynamically configured physics-informed neural network in topology optimization applications, Comput. Methods Appl. Mech. Eng. 426 (2024) 1-23.

[18] G.E. Karniadakis, I.G. Kevrekidis, L. Lu, P. Perdikaris, S. Wang, L. Yang, Physics-informed machine learning, Nat. Rev. Phys. 3 (6) (2021) 422-440.

[19] S. Cai, Z. Wang, S. Wang, P. Perdikaris, G.E. Karniadakis, Physics-informed neural networks for heat transfer problems, J. Heat Transfer 143 (6) (2021).

[20] Z. Mao, A.D. Jagtap, G.E. Karniadakis, Physics-informed neural networks for high-speed flows, Comput. Methods Appl. Mech. Eng. 360 (2020) 1-26.

[21] C. Cheng, H. Meng, Y.-Z. Li, G.-T. Zhang, Deep learning based on PINN for solving 2 DOF vortex induced vibration of cylinder, Ocean Eng. 240 (2021) 1-13.

[22] L. Wang, G. Liu, G. Wang, K. Zhang, M-PINN: a mesh-based physics-informed neural network for linear elastic problems in solid mechanics, Int. J. Numer. Methods Eng. 125 (9) (2024) 1-17.

[23] Y. Chen, L. Lu, G.E. Karniadakis, L. Dal Negro, Physics-informed neural networks for inverse problems in nano-optics and metamaterials, Opt. Express 28 (8) (2020) 11618-11633.

[24] L. Lu, R. Pestourie, W. Yao, Z. Wang, F. Verdugo, S.G. Johnson, Physics-informed neural networks with hard constraints for inverse design, SIAM J. Sci. Comput. 43 (6) (2021) B1105-B1132.

[25] H. Jeong, C. Battuwatta-Gamage, J. Bai, Y.M. Xie, C. Rathnayaka, Y. Zhou, Y. Gu, A complete physics-informed neural network-based framework for structural topology optimization, Comput. Methods Appl. Mech. Eng. 417 (2023) 1-22.

[26] H. Jeong, J. Bai, C. Batuwatta-Gamage, Z.J. Wegert, C.N. Mallon, V.J. Challis, Y. Gui, Y. Gu, Fourier feature embedded physics-informed neural network-based topology optimization (FF-PINNTO) framework for geometrically nonlinear structures, Comput. Methods Appl. Mech. Eng. 446 (2025) 118244.

[27] Y. Zhu, W. Chen, J. Deng, X. Bian, Physics-informed neural networks for hidden boundary detection and flow field reconstruction, arXiv preprint arXiv:2503.24074 (2025).

[28] S. Mowlavi, K. Kamrin, Topology optimization with physics-informed neural networks: application to noninvasive detection of hidden geometries, arXiv preprint arXiv:2303.09280 (2023).

[29] Y.H. Choi, G.H. Yoon, A new density filter for pipes for fluid topology optimization, J. Fluid Mech. 986 (2024) A9.

[30] Y. Zhou, T. Nomura, E.M. Dede, K. Saitou, Topology optimization with wall thickness and piecewise developability constraints for foldable shape-changing structures, Struct. Multidiscip. Optim. 65 (4) (2022) 118.

[31] Y. Wang, Z. Kang, A level set method for shape and topology optimization of coated structures, Comput. Methods Appl. Mech. Eng. 329 (2018) 553-574.

[32] G.H. Yoon, B. Yi, A new coating filter of coated structure for topology optimization, Struct. Multidiscip. Optim. 60 (4) (2019) 1527-1544.

[33] W. Zhang, J. Zhang, X. Guo, Lagrangian description based topology optimization-a revival of shape optimization, J. Appl. Mech. 83 (4) (2016) 1-16.

[34] Z. Li, H. Xu, S. Zhang, A comprehensive review of explicit topology optimization based on moving morphable components (MMC) method, Arch. Comput. Methods Eng. 31 (5) (2024) 2507-2536.

[35] X. Lei, C. Liu, Z. Du, W. Zhang, X. Guo, Machine learning-driven real-time topology optimization under moving morphable component-based framework, J. Appl. Mech. 86 (1) (2019) 1-9.

[36] Z. Du, T. Cui, C. Liu, W. Zhang, Y. Guo, X. Guo, An efficient and easy-to-extend matlab code of the moving morphable component (MMC) method for three-dimensional topology optimization, Struct. Multidiscip. Optim. 65 (5) (2022) 158.

[37] A. Lotfalian, P. Esmaeilpour, G.H. Yoon, M. Takalloozadeh, Integrating moving morphable components and plastic layout optimization: a two-stage approach for enhanced structural topology optimization, Struct. Multidiscip. Optim. 68 (2) (2025) 1-19.

[38] G. Raze, J. Morlier, Explicit topology optimization through moving node approach: beam elements recognition, arXiv preprint arXiv:2103.08347 (2021).

[39] X. Guo, J. Zhou, W. Zhang, Z. Du, C. Liu, Y. Liu, Self-supporting structure design in additive manufacturing through explicit topology optimization, Comput. Methods Appl. Mech. Eng. 323 (2017) 27-63.

[40] T.T. Nguyen, J.A. Bærentzen, O. Sigmund, N. Aage, Efficient hybrid topology and shape optimization combining implicit and explicit design representations, Struct. Multidiscip. Optim. 62 (2020) 1061-1069.

[41] N.P. Van Dijk, K. Maute, M. Langelaar, F. Van Keulen, Level-set methods for structural topology optimization: a review, Struct. Multidiscip. Optim. 48 (3) (2013) 437-472.

[42] M. Tang, Z. Xin, L. Wang, Physics-Informed neural network for level set method in vapor condensation, Int. J. Heat Fluid Flow 110 (2024) 109651.

[43] S. Osher, R. Fedkiw, S. Osher, R. Fedkiw, Constructing signed distance functions, Level Set Methods Dyn. Implicit Surf. 153 (2003) 63-74.

[44] K. He, X. Zhang, S. Ren, J. Sun, Delving deep into rectifiers: surpassing human-level performance on imagenet classification, in: Proceedings of the IEEE International Conference on Computer Cision, 2015, pp. 1026-1034.

[45] N. Rahaman, A. Baratin, D. Arpit, F. Draxler, M. Lin, F. Hamprecht, Y. Bengio, A. Courville, On the spectral bias of neural networks, in: International Conference on Machine Learning, PMLR, 2019, pp. 5301-5310.

[46] Z.-Q.J. Xu, Y. Zhang, T. Luo, Overview frequency principle/spectral bias in deep learning, Commun. Appl. Math. Comput. 7 (2024) 1-38.

[47] D. Lin, S. Kenjeres, Towards fast and reliable estimations of 3D pressure, velocity and wall shear stress in aortic blood flow: CFD-based machine learning approach, Comput. Biol. Med. 191 (2025) 110-137.

[48] M. Pingjian, Z. Wenping, Numerical simulation of low Reynolds number fluid-structure interaction with immersed boundary method, Chin. J. Aeronaut. 22 (5) (2009) 480-485.

[49] W.K. George, Asymptotic effect of initial and upstream conditions on turbulence J. Fluids Eng. 134 (2012) 1-27.

[50] G.L. Morrison, K.R. Hall, M.L. Macek, L.M. Ihfe, R.E. DeOtte, Jr, J.E. Hauglie, Upstream velocity profile effects on orifice flowmeters, Flow Meas. Instrum. 5 (2) (1994) 87-92.

[51] Y.-P. Tsai, Impact of flow velocity on the dynamic behaviour of biofilm bacteria, Biofouling 21 (5-6) (2005) 267-277.

[52] Y. Zhou, T. See, S. Zhong, Z. Liu, L. Li, A massive reduction of dust particle adhesion in a cyclone by the introduction of a wedge, Proc. Inst. Mech. Eng. Part C: J. Mech. Eng. Sci. 232 (17) (2018) 3102-3114.

[53] Y. Zhao, G. Guo, W. Zuo, MATLAB implementations for 3D geometrically nonlinear topology optimization: 230-line code for SIMP method and 280-line code for MMB method, Struct. Multidiscip. Optim. 66 (7) (2023) 146.

[54] S. Bonfanti, S. Hiemer, R. Zulkarnain, R. Guerra, M. Zaiser, S. Zapperi, Computational design of mechanical metamaterials, Nat. Comput. Sci. 4 (8) (2024) 574-583.

[55] F. He, H. An, M. Ghisalberti, S. Draper, C. Ren, P. Branson, L. Cheng, Obstacle arrangement can control flows through porous obstructions, J. Fluid Mech. 992 (2024) 1-42.

[56] H. You, R.O. Tinoco, Characterization of porous in-stream structures to assess their implications on flow dynamics and sediment transport, J. Geophys. Res. Earth Surf. 130 (3) (2025) 1-22.

[57] E. Ronglan, A.P. Rubio, A.O. da Silva, D. Fan, J.L. Gair, Jr, P.M. Stathatou, C. Bastidas, E. Strand, F.J. del Aguila, N. Gershenfeld, et al., Architected materials for artificial reefs to increase storm energy dissipation, PNAS Nexus 3 (3) (2024) 101.

[58] D. Gottlieb, C.-W. Shu, On the Gibbs phenomenon and its resolution, SIAM Rev. 39 (4) (1997) 644-668.