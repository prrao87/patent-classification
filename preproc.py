"""
Code to preprocess raw XML data downloaded from https://bulkdata.uspto.gov/

* Extracts valid XML blocks from each large unzipped file so they can be parsed in
  Python
* Writes out text data (from abstracts/titles) & class labels to a JSON file
"""
import argparse
import json
from glob import glob
from pathlib import Path
from typing import Dict, List
from xml.etree.ElementTree import Element, parse

from tqdm import tqdm


def separate_xml(infile: str, outfile_path: str = "clean_data") -> None:
    """
    Copies individual XML chunks that each contain valid, parsable XML trees and
    writes to new files.
    """
    clean_dir = Path.cwd() / "clean_data"
    clean_dir.mkdir(exist_ok=True, parents=True)
    # os.makedirs(outfile_path, exist_ok=True)
    counter = 1
    outfile = open(Path.cwd() / outfile_path / f"{counter}.xml", "w")
    start_pattern = """<?xml version="1.0" encoding="UTF-8"?>"""
    end_pattern = """</us-patent-grant>"""

    with open(infile) as infile:
        copy = False
        for line in infile:
            if line.startswith(start_pattern):
                copy = True
                outfile.write(start_pattern + "\n")
            elif line.startswith(end_pattern):
                copy = False
                outfile.write(end_pattern + "\n")
                outfile.close()
                counter += 1
                outfile = open(Path.cwd() / outfile_path / f"{counter}.xml", "w")
            elif copy:
                outfile.write(line)
    outfile.close()


def get_xml_file_list(xml_path: str) -> List[str]:
    """
    Return a list of XML file names within a given directory
    """
    xml_files = [f for f in glob(f"{xml_path}/*.xml")]
    return xml_files


def get_subelement(root: Element, element: str, subelement: str) -> str:
    """
    Iterate through specific subelements of an XML root object
    """
    try:
        # Extract the first item in the iterator if it exists
        elem = next(root.iter(element))
        subnode = elem.find(subelement).text
    except StopIteration:
        # If no iterator is found, return None
        subnode = None
    return subnode


def get_element(root: Element, element: str) -> str:
    """
    Iterate through specific elements of an XML root object
    """
    try:
        # Extract the first item in the iterator if it exists
        elem = next(root.iter(element))
        node = elem.text
    except StopIteration:
        # If no iterator is found, return None
        node = None
    return node


def extract_abstracts_and_titles(xml_file: str) -> Dict[str, str]:
    """
    Parse a valid XML tree structure and obtain relevant patent data for the purposes of
    training a machine learning model.
    """
    try:
        tree = parse(xml_file)
        root = tree.getroot()

        doc_id = get_subelement(root, "document-id", "doc-number")
        section_label = get_subelement(root, "classification-ipcr", "section")
        abstract = get_subelement(root, "abstract", "p")
        title = get_element(root, "invention-title")

        # Keep the data only if we have valid text in the title and abstract
        if all((section_label, abstract, title)):
            data = {
                "doc_id": doc_id,
                "title": title,
                "abstract": abstract,
                "label": section_label,
            }
        else:
            data = {}
    except Exception as e:
        # TODO: Use logger instead of print statements
        print(f"{e}. Missing element in `{xml_file}`. Ignoring...")
        data = {}
    return data


def write_json_data(xml_files: List[str]) -> None:
    """
    Iterate through each XML file and write data to a JSON file
    """
    fname = Path(raw_xml_file).stem
    with open(Path.cwd() / f"data_{fname}.jsonl", "w") as f:
        for xml_file in tqdm(xml_files):
            data = extract_abstracts_and_titles(xml_file)
            if data:
                f.write(json.dumps(data) + "\n")


def main(raw_xml_file: str) -> None:
    # path to output clean XML data
    clean_data_path = "clean_data"
    # separate into valid XML chunks and write to individual files
    separate_xml(raw_xml_file, clean_data_path)
    # Read in XML files and write to JSON
    xml_files = get_xml_file_list(clean_data_path)
    write_json_data(xml_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("XML to JSONL converter")
    parser.add_argument("--file", "-f", required=True, help="Path to raw XML file")
    args = vars(parser.parse_args())
    raw_xml_file = args["file"]
    main(raw_xml_file)
