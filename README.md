# sarcasm_detection
- Run the following command in the main directory : sarcasm_detection/ \
`pip install -r requirements.txt --upgrade`
- Folder structure :
  - Sarcasm_detection
    - data
      - dataset (tsv files of raw and cleaned data)
      - source (helper/config type files)
        - common_abbreviations.json
        - data_urls.txt
    - helpers
      - data_collection.py
 - Navigate to 'helpers' folder.
 - To see a test run on the sample tsv provided in the dataset, run `python data_collection.py`
 - To run it on real data, i.e, all the tsv files provided in the dataset, run `python data_collection.py run > log.txt` . This may take several hours
 as each tsv file is around 4GB large.
 
