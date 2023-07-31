# Distractor Selector

1. Installation

Clone this repository and install requirements

```bash
git clone https://github.com/nicklogin/DistractorSelector
python -m pip install -r requirements.txt
python -m nltk.downloader punkt
```

2. Installation via Docker

You can install DistractorSelector image from Dockerhub:

```bash
docker pull niklogin/disselector:latest
```

You can also build a Docker image from source after cloning this repository:

```bash
docker build . -t disselector:latest
```

3. Usage via terminal/command line

DistractorSelector accepts a CSV file with sentences as an input and outputs a CSV file with distractors.

The input file must contian the following fields (example - gold_standard/gold_standard_input.csv):

<b>Masked sentence</b> - The sentence where the target word is replaces with [MASK] token

<b>Right answer</b> - The target word

Run this command to get output of DistractorSelector:

```bash
python -m distractor_generator --
```

DistractorSelector accepts the following set of CLI arguments:

<table>
    <th>
        <tr>
            <td>Argument</td>
            <td>Description</td>
            <td>Default value</td>
        </tr>
    </th>
    <tr>
        <td>--filename</td>
        <td>Path to input file</td>
        <td>gold_standard/gold_standard_input.csv</td>
    </tr>
    <tr>
        <td>--output_filename</td>
        <td>Path to output file</td>
        <td>data/gold_standard_output.csv</td>
    </tr>
    <tr>
        <td>--sep</td>
        <td>Field delimiter in a CSV file</td>
        <td>;</td>
    </tr>
    <tr>
        <td>--index_col</td>
        <td>The name of the index column in a CSV file</td>
        <td>None</td>
    </tr>
    <tr>
        <td>--n</td>
        <td>Number of distractors on the classifier input</td>
        <td>20</td>
    </tr>
    <tr>
        <td>--no-clf</td>
        <td>Do not use classifier</td>
        <td> - </td>
    </tr>
    <tr>
        <td>--clf_path</td>
        <td>Path to file with the saved classifier model</td>
        <td>XGBAllFeats/clf.pkl</td>
    </tr>
    <tr>
        <td>--cols_path</td>
        <td>Path to file with feature names to be used with the classifier</td>
        <td>XGBAllFeats/cols.json</td>
    </tr>
</table>

4. Usage via a WebAPI

Execute this command in command line/terminal to run the API of DistractorSelector:

```bash
python -m api
```

The documentation will be available at http://localhost:5000/docs
