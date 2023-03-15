# Setting up a FloodMapper Processing VM on GCP

ML4Floods and FloodMapper are designed to run on a computer with
machine learning accelerator (e.g., GPU, TPU etc). The core requirement
is a Python environment capable of running the PyTorch machine
learning framework.

The processing machine can be a local computer with a UNIX-like
operating system (e.g., Linux or MacOS), but it is convenient to
create a dedicated a virtual machine (VM) on GCP.

## Setting up a Deep Learning instance

GCP provides specialised virtual machines that are pre-configured for
Deep Learning workflows. They are set up with a GPU and already have
most of the necessary software installed (e.g., NVIDIA CUDA, Python,
PyTorch etc.). To set up a Deep Learning VM, do the following:

 1. Navigate to the 'Compute Engine > [VM
    Instances](https://console.cloud.google.com/compute/instances)'
    page on GCP Console and click on **Marketplace** in the sidebar.

 1. Search for 'pytorch' and click on the **Deep Learning VM** option.

 1. Click on **Launch** to bring up the configuration page.

 1. Click **Enable** if you are asked about missing APIs.

 1. On the configuration page:
     * Provide a Deployment Name like 'floodmapper-dl'.
     * Choose a local zone (e.g., 'australia-southeast1-a').
     * Choose 'General Purpose' as a machine mype.
     * Choose a 'N1' series machine like 'n1-highmem-4'.
     * Choose a single 'NVIDIA T4' GPU.
     * As a Framework, choose 'PyTorch 1.13'.
     * Enable 'Install NVIDIA GPU drivers automatically'.
     * Enable 'Access to JupyterLab via URL'.
     * Accept the default boot disk, network interface and T&Cs.
     * Click **Deploy**. The VM will take a few minutes to be created.

After the VM has been deployed, click on the **tensorflow.jinja** text
to bring up the 'Getting Started' information. There is a direct link
under **SSH** to bring up a terminal in a browser. Clicking this will
log you onto the machine **under your current Google username**, in a
directory called ```/home/<username>```. Your account has *sudo*
access to install software, which we can do with some simple
commands. Just now we will install the command line client to connect
to the PostgreSQL server:

```
# Install the PostgreSQL client
sudo apt install postgresql-client
```

This terminal window also has convenient buttons to upload and
download files directly to the VM. Now is a good time to upload the
JSON-format key file associated with the 'bucket_access' service
account we created earlier.

 1. Click **Upload File**.

 1. Click **Choose Files**, navigate to where you downloaded the key
 file and click the **Upload Files** button.

The file will appear in the current directory (confirm by running
```ls``` in the terminal). If the upload processes throws an error,
restart the SSH terminal and try again immediately.

At this point we will download a copy of the FloodMapper repo so we
can set up the Python environment. Execute the following in the
SSH terminal:

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
     This will take a a few minutes to run - now is a good time to
     take a tea break!

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


## Accessing your VM through Jupyter Lab in a browser

Jupyter Lab gives you access to Jupyter notebooks, a Python terminal
and standard command line shell. Using the GCP Console, you can launch
a Jupyter Lab session that is connected to your VM:

 1. Navigate to 'Vertex AI >
    [Workbench](https://console.cloud.google.com/vertex-ai/workbench)'
    in the GCP Console.
 1. Click the **Open JupyterLab** link next to your VM instance name.

 1. JupyterLab will start up!

**NB:** JupyterLab does not give you access to the VM via your Google
user account, but through a special ```/home/jupyter``` account set up
for the Jupyter server. This means that you won’t see your own Google
user account home directory, but a shared ```jupyter``` one for anyone
that accesses the notebook. We will do all of our processing in this
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

## Downloading FloodMapper for production use

Since we will be running the FloodMapper system through the Jupyter
account, we need to download a new copy of the FloodMapper codebase
under ```/home/jupyter```.


 1. Download the FloodMapper repo from Github:
     ```
     # Fetch the FloodMapper code
     git clone https://github.com/spaceml-org/floodmapper.git
     ```
 1. Now is also a good time to copy the GCP access key from your user directory:
     ```
     # Copy the key file to the Jupyter account
     cp ../<username>/<key_file.json> .
     ```

Close the JupyterLab session by clicking **File > Shut Down**. If you
re-open JupyterLab, you should see the option to open Python notebooks
under the `(floodmapper)` environment.


The processing machine is almost ready to use, aside from enabling
database and bucket access. We will set these up in the next steps.

---

## NEXT: [Configuring the FloodMapper system](02c_SETUP_CONFIGURATION.md)
