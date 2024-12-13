# sparse-vllm

> This is the repository for the Harvard CS243 (Fall 2024) final project: *SpvLLM: Efficient KV Cache Sparsification in vLLM*.

## Table of Contents

- [AWS Setup](#aws-setup)
- [Virtual Machine Setup](#virtual-machine-setup)
- [VLLM Setup](#vllm-setup)
- [Experiments](#experiments)

## AWS Setup

- Go to the service quotas dashboard then Amazon EC2 (https://us-east-2.console.aws.amazon.com/servicequotas/home/services/ec2/quotas, change `us-east-2` to your region). Search for "Running On-Demand G and VT instances", then request increase quota to at least 8 vCPUs. *This needs to be done only once.*
- Go to [EC2 dashboard](https://us-east-2.console.aws.amazon.com/ec2/home) (choose the correct region), go to instances in the sidebar then launch instance. Give it a name, choose "Ubuntu Server 22.04 LTS (HVM), SSD Volume Type" for the Amazon Machine Image, and use the default x86_64 architecture. For instance type, search for `g4dn.2xlarge` or larger g-family or p-family instances with NVIDIA GPUs. Select a key pair (see [course infra page](https://github.com/minlanyu/cs243-site/blob/fall2024/infra.md)). Select an existing security group that allows all IPv4 (see course warmup project), or let it create a new security group. Configure storage to 60GiB, and leave other configurations unchanged. Launch the instance.
- Connect to the instance with your key pair. There is a connect instruction if you click on your instance in AWS and click connect.

<details>
<summary>CHMOD 400 on Windows</summary>
<p>

Note that in the connection instructions it requires `chmod 400` to make the key not publicly available. To achieve the equivalent on Windows, do the following in Powershell:

```cmd
icacls.exe "/PATH/TO/PEM/FILE" /reset
icacls.exe "/PATH/TO/PEM/FILE" /grant:r "$($env:username):(r)"
icacls.exe "/PATH/TO/PEM/FILE" /inheritance:r
```

</p>
</details>

<details>
<summary>Connecting VSCode to the server</summary>
<p>

To connect with VSCode, add the following to the SSH configuration file:

```
Host aws-ec2-cs243
    HostName ec2-3-17-72-136.us-east-2.compute.amazonaws.com
    User ubuntu
    IdentityFile /PATH/TO/PEM/FILE
```

</p>
</details>

## Virtual Machine Setup

Now inside the server, first make sure system dependencies are up-to-date and install other needed packages:

```bash
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y ccache
which ccache  # Confirm that ccache is discoverable
```

Install micromamba:

```bash
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh  # Follow the prompts (Enter, yes, Enter, Enter)
eval "$(/home/ubuntu/miniforge3/bin/conda shell.bash hook)"
conda init
rm Miniforge3-Linux-x86_64.sh
```

Install CUDA toolkit:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
rm cuda-keyring_1.1-1_all.deb
```

Install CUDA driver:

```bash
sudo apt-get install -y nvidia-open
sudo reboot  # You would need to reconnect to the instance after a few seconds
nvidia-smi  # Check that it worked
```

## VLLM Setup

Set enviroment variables to (1) let vLLM discover cuda, (2) reduce the number of compilation jobs that run simultaneously (8 jobs and 32GiB memory sometimes gives OOM, 6 jobs also gives OOM in rare occasions, and 4 jobs is currently safe by my experience), (3) use `xformers` backend (and thus disable the `vllm-flash-attn` build from source), and (4) disable build of `_moe_C` because we are at least not using those for now. You may add the following to `~/.bashrc` then do `source ~/.bashrc` to reload the configurations:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:$PATH"
export MAX_JOBS=4
export VLLM_ATTENTION_BACKEND=XFORMERS
export DISABLE_MOE_BUILD=1
```

You can confirm that vLLM can discover `CUDA_HOME` and make sure `nvcc` compiler is in `PATH` by:

```bash
nvcc --version  # Check nvcc
${CUDA_HOME}/bin/nvcc --version  # Check nvcc is in CUDA_HOME
```

Configure `git`:

```bash
git config --global core.editor vim  # Or other editors you prefer
git config --global user.name YOUR_USERNAME
git config --global user.email YOUR_EMAIL
```

Clone the repository and build vLLM. Note that you should currently be in the conda base environment that is by default activated.

```bash
git clone https://github.com/Charlie-XIAO/sparse-vllm.git  # Or use ssh
cd sparse-vllm
pip install -v -e .  # This can take very long
python -c "import vllm; print(vllm.__version__)"  # Validate the installation
```

Finally install the dependencies.

```bash
pip install -r requirements-dev.txt
pip install seaborn  # Needed for experiments
```

For linting, run:

```bash
./format.sh
```

## Experiments

Make sure you are in the current directory. First download the ShareGPT dataset:

```bash
wget -O ./benchmarks/sharegpt.json https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

Then to reproduce our experiment results, simply run:

```bash
./run243.sh
```

The plots will be available under the `./cs243/plots` directory.
