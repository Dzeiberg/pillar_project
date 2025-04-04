# IGVF Coding Variant Focus Group Pillar Project
## Installation

To set up the project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Dzeiberg/pillar_project.git
    ```

2. Navigate to the project directory:
    ```bash
    cd pillar_project
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
## Usage

### Downloading Datasets

To download the datasets required for this project, follow these steps:

1. Visit the Zenodo dataset link: [https://zenodo.org/records/15149879](https://zenodo.org/records/15149879)

2. Request access and download the dataset files to your local machine.

3. Extract the downloaded files (if compressed) into the `data/` directory within the project folder:
    ```bash
    mkdir -p data
    tar -xvf downloaded_file.tar.gz -C data/
    ```

4. Verify that the dataset files are correctly placed in the `data/` directory.

### Running `run_single_fit`

To run a single calibration 
1. Ensure that all dependencies are installed as described in the **Installation** section.

2. Run the `run_single_fit` function using Python:
    ```bash
    mkdir -p fit_results/
    python run_fits.py scoreset_name data/scoreset_name.pkl fit_results/
    ```

For additional options or help, you can run:
```bash
python run_fits.py --help
```