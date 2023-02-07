# Processing on a GCP Virtual Machine

## Setting up a Deep Learning instance

 1. Navigate to the 'Compute Engine > [VM
    Instances](https://console.cloud.google.com/compute/instances)'
    page on GCP Console and click on **Marketplace** in the sidebar.

 1. Search for 'pytorch' and click on the **Deep Learning VM**.

 1. Click on **Launch** to bring up the configuration page.

 1. Click **Enable** if you are asked about missing APIs.

 1. On the configuration page:
     * Provide a Deployment Name like 'floodmapper-dl'.
     * Choose a 'N1' series machine like 'n1-highmem-4'.
     * Choose a single 'NVIDIA T4' GPU.
     * As a Framework, choose 'PyTorch 1.13'.
     * Enable 'Install NVIDIA GPU drivers automatically'.
     * Enable 'Access to JupyterLab via URL'.
     * Accept the default boot disk, network interface and T&Cs.
     * Click **Deploy**.

After the VM has been deployed, click on the **tensorflow.jinja** text
to bring up the 'Getting Started' information. There is a direct link
under **SSH** to bring up a terminal in a browser. Clicking this will
log you onto the machine under your current Google username, in a
directory called ```/home/<username>```. Your account has *sudo*
access to install software, which we can do with some simple commands:

```
# Install the PostgreSQL client
sudo apt install postgresql-client
```

This terminal window also has convenient buttons to upload and
download files directly to the VM.

At this point we will download a copy of the FloodMapper repo, so we
can setup the Python environment. Execute the following in the
terminal:

 1. Download the FloodMapper repo from Github:
     ```
     # Fetch the FloodMapper code
     git clone https://github.com/spaceml-org/floodmapper.git
     ```

 1. Create a new Python environment to run the code:
     ```
     cd floodmapper/tutorial
     conda env create -f floodmapper.yml -n floodmapper
     ```

 1. Activate the new environment as a test:
     ```
     # Activate the environment
     conda activate floodmapper
     ```

Note: if you ever need to update the conda environment with new
packages, simply edit the ```floodmapper.yml``` file and run the
following command in the SSH terminal:

```
conda env update --name floodmapper --file floodmapper.yml --prune
```


However, most of our interaction with the processing machine will be
through [Jupyter Lab](https://jupyter.org/), which we will start now.


## Accessing your VM through Jupyter Lab

Jupyter Lab gives you access to Jupyter notebooks, a Python terminal
and standard command line shell. You launch a Jupyter Lab session
connected to your VM through the GCP Console:

 1. Navigate to 'Vertex AI >
    [Workbench](https://console.cloud.google.com/vertex-ai/workbench)'
    in the GCP Console.
 1. Click the **Open JupyterLab** link next to your VM instance name.

 1. JupyterLab will start up!

Note that JupyterLab does not give you access to the VM via your
Google user account, but through a special ```/home/jupyter``` account
set up for the server. This means that you won’t see your own google
user account home directory, but a shared jupyter one for anyone that
accesses the notebook. We will do all of our processing in this
folder.

## Registering the FloodMapper environment

We can let JupyterLab know about our new Python ```floodmapper```
environment by registering the new processing kernel.


 1. Open a JupyterLab terminal by clicking on the icon.

 1. Register the kernel with Jupyter:

     ```
     # Register with Jupyter
     python -m ipykernel install --user --name floodmapper \
     --display-name "Python (floodmapper)"
     ```

 1. Test that we can activate the environment:
     ```
     # Activate the environment
     conda activate floodmapper
     ```

The Processing Machine is almost ready to use, aside from enabling
database and bucket access. We will set these up in the next steps.