# to converte txt file to csv file (the DataSet)
import csv
import json
from collections import namedtuple
import re

class ConvertToCSV:

    # from txt file
    def convert_to_csv(name):
                
        # Open the input text file
        with open(name, 'r') as in_file:
            # Strip leading/trailing whitespace from each line
            stripped = (line.strip() for line in in_file)
            
            # Split each line on the tab character and filter out empty lines
            lines = (line.split("\t") for line in stripped if line)
            
            # Open the output CSV file
            with open(name+'.csv', 'w', newline='') as out_file:
                writer = csv.writer(out_file)
                
                # Write the header row
                writer.writerow(('num', 'doc'))
                
                # Write the data rows, skipping any that are empty
                for line in lines:
                    if line:  # Check if the line is not empty
                        writer.writerow(line)

    # from jsonl file
    # def jsonl_to_csv(input_file, output_file):
    #     TipOfTheTongueDoc = namedtuple('TipOfTheTongueDoc', ['page_title', 'page_source', 'wikidata_id', 'wikidata_classes', 'text', 'sections', 'infoboxes', 'doc_id'])

    #     with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
    #         data = [json.loads(line) for line in f]

    #     with open(output_file, 'w', newline='', encoding='utf-8', errors='ignore') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(['page_title', 'page_source', 'wikidata_id', 'wikidata_classes', 'text', 'sections', 'infoboxes', 'doc_id'])

    #         for item in data:
    #             wikidata_classes = ','.join([wc[1] for wc in item['wikidata_classes']])

    #             # Process text
    #             text = re.sub(r'<[^>]+>', ' ', item['text'])  # Remove HTML tags
    #             text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    #             text = ' '.join(text.split()[:70])  # Take only the first 70 words

    #             # Process page_source
    #             page_source = re.sub(r'[^a-zA-Z0-9\s]', ' ', item['page_source'])
    #             page_source = re.sub(r'\||\{|\}', ' ', page_source)  # Remove additional special characters
    #             page_source = re.sub(r'\s+', ' ', page_source)
    #             page_source = ' '.join(page_source.split()[:70])

    #             # Process infoboxes
    #             infoboxes = [re.sub(r'[^a-zA-Z0-9\s]', '', str(box)) for box in item['infoboxes']]
    #             infoboxes = [re.sub(r'\||\{|\}|\"', '', str(box)) for box in infoboxes]  # Remove additional special characters and double quotes
    #             infoboxes = json.dumps(infoboxes)

    #             # Process sections
    #             sections = ','.join(item['sections'].keys())

    #             doc = TipOfTheTongueDoc(
    #                 page_title=item['page_title'],
    #                 page_source=page_source,
    #                 wikidata_id=item['wikidata_id'],
    #                 wikidata_classes=wikidata_classes,
    #                 text=text,
    #                 sections=sections,
    #                 infoboxes=infoboxes,
    #                 doc_id=item['doc_id']
    #             )
    #             writer.writerow([
    #                 doc.page_title,
    #                 doc.page_source,
    #                 doc.wikidata_id,
    #                 doc.wikidata_classes,
    #                 doc.text,
    #                 doc.sections,
    #                 doc.infoboxes,
    #                 doc.doc_id
    #             ])

    # Example usage
    # jsonl_to_csv('trecPart.jsonl', 'trec-part1000.csv')
    # to do: 
    # jsonl_to_csv('C:\\Users\\DELL\\Desktop\\IR_Omama\\TREC-TOT\\corpus.jsonl', 'trec-tot.csv')

    def convert_from_tsv(self, name):
        # Open the input TSV file
        with open(name, 'r') as tsv_file:
            # Read the TSV file
            tsv_data = tsv_file.readlines()

        # Open the output CSV file
        with open('recreation-collection.csv', 'w', newline='') as csv_file:
            # Create a CSV writer
            csv_writer = csv.writer(csv_file)

            # Write the header row
            csv_writer.writerow(["num", "doc"])

            # Iterate through the TSV data and write to the CSV file
            for line in tsv_data:
                # Split the line by tab
                fields = line.strip().split('\t')

                # Create a new row with the "num" and "doc" columns
                row = [fields[0], ' '.join(fields[1:])]

                # Write the row to the CSV file
                csv_writer.writerow(row)